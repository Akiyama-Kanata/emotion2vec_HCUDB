import argparse

import torch

try:
    from vad_downstream.data import EMOTION_CLASS_LABELS, EMOTION_CLASS_NAMES_JA
    from vad_downstream.inference import (
        build_audio_encoder,
        load_wav_16khz_mono,
        resolve_device,
        validate_checkpoint_args,
        write_json,
    )
    from vad_downstream.model import Emotion2vecParallelEmotionVADClassifier
    from vad_downstream.parallel_training import DOMINANCE_STATUSES
except ModuleNotFoundError:
    from data import EMOTION_CLASS_LABELS, EMOTION_CLASS_NAMES_JA
    from inference import (
        build_audio_encoder,
        load_wav_16khz_mono,
        resolve_device,
        validate_checkpoint_args,
        write_json,
    )
    from model import Emotion2vecParallelEmotionVADClassifier
    from parallel_training import DOMINANCE_STATUSES


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run single-WAV independent emotion and V/A/D inference."
    )
    parser.add_argument("--wav", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv=None, encoder_factory=None):
    args = get_parser().parse_args(argv)
    encoder = encoder_factory(args) if encoder_factory is not None else None
    payload = run_inference(
        args.wav,
        args.classifier_checkpoint,
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        encoder=encoder,
    )
    write_json(payload, args.output)
    return payload


def run_inference(
    wav_path,
    classifier_checkpoint,
    model_dir=None,
    checkpoint=None,
    device="auto",
    encoder=None,
):
    validate_checkpoint_args(model_dir=model_dir, checkpoint=checkpoint)
    torch_device = resolve_device(device)
    payload = torch.load(classifier_checkpoint, map_location=torch_device)
    config = read_parallel_config(payload)
    encoder = encoder if encoder is not None else build_audio_encoder(
        model_dir=model_dir, checkpoint=checkpoint, device=torch_device
    )
    model = Emotion2vecParallelEmotionVADClassifier(
        encoder,
        num_classes=config["num_classes"],
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        freeze_encoder=True,
    ).to(torch_device)
    state = _extract_state_dict(payload)
    if any(key.startswith("head.") for key in state):
        state = {
            key[len("head.") :]: value
            for key, value in state.items()
            if key.startswith("head.")
        }
    model.head.load_state_dict(state)
    model.eval()
    wav = load_wav_16khz_mono(wav_path).to(torch_device)
    with torch.no_grad():
        output = model(wav.unsqueeze(0))
    return make_payload(
        wav_path,
        classifier_checkpoint,
        output["logits"][0].cpu(),
        output["vad"][0].cpu(),
        config,
    )


def read_parallel_config(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("classifier checkpoint must be a dict")
    if checkpoint.get("model_type") != "parallel_emotion_vad":
        raise ValueError("checkpoint is not a parallel_emotion_vad model")
    labels = list(checkpoint.get("class_labels", EMOTION_CLASS_LABELS))
    num_classes = int(checkpoint.get("num_classes", len(labels)))
    if len(labels) != num_classes:
        raise ValueError("checkpoint class_labels length does not match num_classes")
    names = list(checkpoint.get("class_names_ja", labels))
    if len(names) != num_classes:
        raise ValueError("checkpoint class_names_ja length does not match num_classes")
    status = checkpoint.get("dominance_status", "untrained")
    if status not in DOMINANCE_STATUSES:
        raise ValueError(f"invalid dominance_status: {status}")
    return {
        "input_dim": int(checkpoint.get("input_dim", 768)),
        "hidden_dim": int(checkpoint.get("hidden_dim", 256)),
        "num_classes": num_classes,
        "class_labels": labels,
        "class_names_ja": names,
        "dominance_status": status,
        "vad_label_counts": checkpoint.get("vad_label_counts", {}),
        "supervised_dimensions": checkpoint.get("supervised_dimensions", []),
    }


def make_payload(wav_path, checkpoint_path, logits, vad, config):
    probabilities = torch.softmax(logits, dim=0)
    predicted = int(probabilities.argmax().item())
    labels = config["class_labels"]
    status = config["dominance_status"]
    result = {
        "wav": str(wav_path),
        "classifier_checkpoint": str(checkpoint_path),
        "model_type": "parallel_emotion_vad",
        "class_labels": labels,
        "prediction": {
            "index": predicted,
            "code": labels[predicted],
            "name_ja": config["class_names_ja"][predicted],
        },
        "probabilities": {
            label: float(probabilities[index].item())
            for index, label in enumerate(labels)
        },
        "logits": {
            label: float(logits[index].item()) for index, label in enumerate(labels)
        },
        "vad": {
            "valence": {"value": float(vad[0].item()), "status": "trained"},
            "arousal": {"value": float(vad[1].item()), "status": "trained"},
            "dominance": {"value": float(vad[2].item()), "status": status},
        },
        "vad_label_counts": config["vad_label_counts"],
        "supervised_dimensions": config["supervised_dimensions"],
    }
    if status == "untrained":
        result["warning"] = (
            "Dominance is emitted numerically but its head has no supervised "
            "training and is not a learned dominance estimate."
        )
    return result


def _extract_state_dict(checkpoint):
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state_dict")
    if any(key.startswith("module.") for key in state):
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
    return state


if __name__ == "__main__":
    main()
