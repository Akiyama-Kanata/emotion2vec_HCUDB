import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from vad_downstream.model import Emotion2vecVADModel
except ModuleNotFoundError:
    from model import Emotion2vecVADModel


LABELS_BY_TARGET_DIM = {
    2: ["valence", "arousal"],
    3: ["valence", "arousal", "dominance"],
}


class Stage1AudioFeatureEncoder(nn.Module):
    """Deterministic placeholder encoder used only for Stage 1 wiring checks."""

    def __init__(self, input_dim=768, frame_size=320):
        super().__init__()
        self.input_dim = input_dim
        self.frame_size = frame_size

    def extract_features(
        self,
        source,
        padding_mask=None,
        mask=False,
        remove_extra_tokens=True,
    ):
        if source.dim() != 2:
            raise ValueError(f"source must be 2D [B, T], got {source.shape}")

        pad_len = (-source.size(1)) % self.frame_size
        if pad_len:
            source = F.pad(source, (0, pad_len))

        frames = source.unfold(dimension=1, size=self.frame_size, step=self.frame_size)
        stats = torch.stack(
            [
                frames.mean(dim=-1),
                frames.std(dim=-1, unbiased=False),
                frames.abs().mean(dim=-1),
                frames.amax(dim=-1),
                frames.amin(dim=-1),
            ],
            dim=-1,
        )
        repeats = math.ceil(self.input_dim / stats.size(-1))
        features = stats.repeat_interleave(repeats, dim=-1)[..., : self.input_dim]
        feature_padding_mask = torch.zeros(
            features.shape[:2], dtype=torch.bool, device=features.device
        )
        return {"x": features, "padding_mask": feature_padding_mask}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run Stage 1 WAV-to-VA/VAD inference and write JSON output."
    )
    parser.add_argument("--wav", required=True, help="Path to a 16kHz mono WAV file.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Reserved for Stage 2 fairseq user module loading.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Reserved for Stage 2 emotion2vec checkpoint loading.",
    )
    parser.add_argument("--target-dim", type=int, choices=(2, 3), required=True)
    parser.add_argument(
        "--head-checkpoint",
        default=None,
        help="Optional VADRegressionHead checkpoint.",
    )
    parser.add_argument(
        "--allow-random-head",
        action="store_true",
        help="Allow JSON output with an untrained randomly initialized head.",
    )
    parser.add_argument("--output", default=None, help="Path to write JSON output.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    return parser


def main(argv=None, encoder_factory=None):
    parser = get_parser()
    args = parser.parse_args(argv)

    encoder = encoder_factory(args) if encoder_factory is not None else None
    payload = run_inference(
        wav_path=args.wav,
        target_dim=args.target_dim,
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        head_checkpoint=args.head_checkpoint,
        allow_random_head=args.allow_random_head,
        device=args.device,
        encoder=encoder,
    )
    write_json(payload, args.output)
    return payload


def run_inference(
    wav_path,
    target_dim,
    model_dir=None,
    checkpoint=None,
    head_checkpoint=None,
    allow_random_head=False,
    device="auto",
    encoder=None,
):
    if target_dim not in LABELS_BY_TARGET_DIM:
        raise ValueError(f"target_dim must be 2 or 3, got {target_dim}")
    if head_checkpoint is None and not allow_random_head:
        raise ValueError(
            "--head-checkpoint is required unless --allow-random-head is set"
        )

    torch_device = resolve_device(device)
    wav = load_wav_16khz_mono(wav_path).to(torch_device)
    encoder = encoder if encoder is not None else build_stage1_encoder(
        model_dir=model_dir,
        checkpoint=checkpoint,
    )
    model = Emotion2vecVADModel(encoder=encoder, target_dim=target_dim)
    model.to(torch_device)
    model.eval()

    if head_checkpoint is not None:
        load_head_checkpoint(model.head, head_checkpoint, torch_device)

    with torch.no_grad():
        prediction = model(wav.unsqueeze(0)).squeeze(0).cpu()

    return make_prediction_payload(
        wav_path=wav_path,
        target_dim=target_dim,
        prediction=prediction,
        head_checkpoint=head_checkpoint,
        random_head=head_checkpoint is None,
    )


def build_stage1_encoder(model_dir=None, checkpoint=None):
    if model_dir is not None or checkpoint is not None:
        raise NotImplementedError(
            "real emotion2vec checkpoint loading is Stage 2; omit --model-dir "
            "and --checkpoint for the Stage 1 placeholder encoder"
        )
    return Stage1AudioFeatureEncoder()


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is not available")
    return torch.device(device)


def load_wav_16khz_mono(path):
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_rate != 16000:
        raise ValueError(f"WAV sample rate must be 16000 Hz, got {sample_rate}: {path}")
    if channels != 1:
        raise ValueError(f"WAV must be mono, got {channels} channels: {path}")

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(
            f"unsupported WAV sample width: {sample_width} bytes in {path}"
        )

    if audio.size == 0:
        raise ValueError(f"WAV has no audio samples: {path}")
    return torch.from_numpy(audio).float()


def load_head_checkpoint(head, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_head_state_dict(checkpoint)
    head.load_state_dict(state_dict)


def extract_head_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("head checkpoint must be a state-dict-like object")

    for key in ("head_state_dict", "state_dict", "model_state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break

    if not isinstance(checkpoint, dict):
        raise ValueError("head checkpoint state_dict must be a dict")

    if any(key.startswith("head.") for key in checkpoint):
        return {
            key[len("head.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("head.")
        }
    return checkpoint


def make_prediction_payload(
    wav_path,
    target_dim,
    prediction,
    head_checkpoint,
    random_head,
):
    labels = LABELS_BY_TARGET_DIM[target_dim]
    values = [float(value) for value in prediction.tolist()]
    return {
        "wav": str(wav_path),
        "target_dim": target_dim,
        "labels": labels,
        "prediction": dict(zip(labels, values)),
        "head_checkpoint": None if head_checkpoint is None else str(head_checkpoint),
        "random_head": bool(random_head),
    }


def write_json(payload, output_path=None):
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output_path is None:
        print(text)
        return

    Path(output_path).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
