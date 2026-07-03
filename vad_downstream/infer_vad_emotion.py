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
    from vad_downstream.model import Emotion2vecVADMediatedClassifier
except ModuleNotFoundError:
    from data import EMOTION_CLASS_LABELS, EMOTION_CLASS_NAMES_JA
    from inference import (
        build_audio_encoder,
        load_wav_16khz_mono,
        resolve_device,
        validate_checkpoint_args,
        write_json,
    )
    from model import Emotion2vecVADMediatedClassifier


VAD_LABELS_BY_TARGET_DIM = {
    2: ["valence", "arousal"],
    3: ["valence", "arousal", "dominance"],
}
CLASS_NAME_JA_BY_LABEL = dict(zip(EMOTION_CLASS_LABELS, EMOTION_CLASS_NAMES_JA))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run WAV -> predicted VAD -> emotion classification JSON inference."
    )
    parser.add_argument("--wav", required=True, help="Path to a 16kHz mono WAV file.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="fairseq user module directory for emotion2vec checkpoint loading.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="emotion2vec checkpoint. Requires --model-dir.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        default=None,
        help="VAD-mediated classifier checkpoint from train_vad_emotion.py.",
    )
    parser.add_argument(
        "--allow-random-model",
        action="store_true",
        help="Allow JSON output with an untrained randomly initialized classifier.",
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        choices=(2, 3),
        default=3,
        help="Target dimension used only with --allow-random-model.",
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
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        allow_random_model=args.allow_random_model,
        target_dim=args.target_dim,
        device=args.device,
        encoder=encoder,
    )
    write_json(payload, args.output)
    return payload


def run_inference(
    wav_path,
    model_dir=None,
    checkpoint=None,
    classifier_checkpoint=None,
    allow_random_model=False,
    target_dim=3,
    device="auto",
    encoder=None,
):
    validate_checkpoint_args(model_dir=model_dir, checkpoint=checkpoint)
    if classifier_checkpoint is None and not allow_random_model:
        raise ValueError(
            "--classifier-checkpoint is required unless --allow-random-model is set"
        )

    torch_device = resolve_device(device)
    wav = load_wav_16khz_mono(wav_path).to(torch_device)
    encoder = encoder if encoder is not None else build_audio_encoder(
        model_dir=model_dir,
        checkpoint=checkpoint,
        device=torch_device,
    )

    checkpoint_payload = None
    if classifier_checkpoint is not None:
        checkpoint_payload = torch.load(classifier_checkpoint, map_location=torch_device)
        config = read_classifier_config(checkpoint_payload)
    else:
        if target_dim not in VAD_LABELS_BY_TARGET_DIM:
            raise ValueError(f"target_dim must be 2 or 3, got {target_dim}")
        config = {
            "target_dim": int(target_dim),
            "input_dim": 768,
            "hidden_dim": 256,
            "class_labels": list(EMOTION_CLASS_LABELS),
            "class_names_ja": list(EMOTION_CLASS_NAMES_JA),
        }

    model = Emotion2vecVADMediatedClassifier(
        encoder=encoder,
        target_dim=config["target_dim"],
        num_classes=len(config["class_labels"]),
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        freeze_encoder=True,
    ).to(torch_device)
    model.eval()

    if checkpoint_payload is not None:
        load_classifier_checkpoint(model.head, checkpoint_payload)

    with torch.no_grad():
        output = model(wav.unsqueeze(0), return_vad=True)
        vad = output["vad"].squeeze(0).detach().cpu()
        logits = output["logits"].squeeze(0).detach().cpu()

    return make_emotion_payload(
        wav_path=wav_path,
        target_dim=config["target_dim"],
        class_labels=config["class_labels"],
        class_names_ja=config["class_names_ja"],
        vad=vad,
        logits=logits,
        classifier=model.head.classifier,
        classifier_checkpoint=classifier_checkpoint,
        random_model=checkpoint_payload is None,
    )


def read_classifier_config(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("classifier checkpoint must be a dict")
    if "target_dim" not in checkpoint:
        raise ValueError("classifier checkpoint is missing target_dim")

    target_dim = int(checkpoint["target_dim"])
    if target_dim not in VAD_LABELS_BY_TARGET_DIM:
        raise ValueError(f"target_dim must be 2 or 3, got {target_dim}")

    class_labels = list(checkpoint.get("class_labels", EMOTION_CLASS_LABELS))
    if class_labels != list(EMOTION_CLASS_LABELS):
        expected = ", ".join(EMOTION_CLASS_LABELS)
        got = ", ".join(class_labels)
        raise ValueError(f"class_labels must be {expected}; got {got}")

    class_names_ja = list(
        checkpoint.get(
            "class_names_ja",
            [CLASS_NAME_JA_BY_LABEL[label] for label in class_labels],
        )
    )
    if len(class_names_ja) != len(class_labels):
        raise ValueError("class_names_ja length must match class_labels length")

    return {
        "target_dim": target_dim,
        "input_dim": int(checkpoint.get("input_dim", 768)),
        "hidden_dim": int(checkpoint.get("hidden_dim", 256)),
        "class_labels": class_labels,
        "class_names_ja": class_names_ja,
    }


def load_classifier_checkpoint(model_head, checkpoint):
    state_dict = extract_model_state_dict(checkpoint)
    model_head.load_state_dict(state_dict)


def extract_model_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("classifier checkpoint must be a dict")

    for key in ("model_state_dict", "state_dict", "head_state_dict"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("classifier checkpoint state_dict must be a dict")

    state_dict = _strip_prefix(state_dict, "module.")
    if any(key.startswith("head.") for key in state_dict):
        return {
            key[len("head.") :]: value
            for key, value in state_dict.items()
            if key.startswith("head.")
        }
    return state_dict


def make_emotion_payload(
    wav_path,
    target_dim,
    class_labels,
    class_names_ja,
    vad,
    logits,
    classifier,
    classifier_checkpoint,
    random_model,
):
    vad_labels = VAD_LABELS_BY_TARGET_DIM[target_dim]
    probabilities = torch.softmax(logits, dim=0)
    ranked_indices = torch.argsort(probabilities, descending=True).tolist()
    prediction_index = int(ranked_indices[0])
    runner_up_index = int(ranked_indices[1])

    weight = classifier.linear.weight.detach().cpu()
    bias = classifier.linear.bias.detach().cpu()

    linear_weights = {}
    contributions = {}
    for class_index, class_label in enumerate(class_labels):
        linear_weights[class_label] = {
            "bias": float(bias[class_index].item()),
        }
        class_contributions = {
            "bias": float(bias[class_index].item()),
        }
        logit_sum = float(bias[class_index].item())

        for dim_index, vad_label in enumerate(vad_labels):
            weight_value = float(weight[class_index, dim_index].item())
            contribution = float((weight[class_index, dim_index] * vad[dim_index]).item())
            linear_weights[class_label][vad_label] = weight_value
            class_contributions[f"w_{vad_label}*{vad_label[0]}"] = contribution
            logit_sum += contribution

        class_contributions["logit_sum"] = logit_sum
        contributions[class_label] = class_contributions

    contrast = make_contrast_payload(
        prediction_index=prediction_index,
        runner_up_index=runner_up_index,
        class_labels=class_labels,
        class_names_ja=class_names_ja,
        vad_labels=vad_labels,
        vad=vad,
        logits=logits,
        probabilities=probabilities,
        weight=weight,
        bias=bias,
    )

    return {
        "wav": str(wav_path),
        "classifier_checkpoint": (
            None if classifier_checkpoint is None else str(classifier_checkpoint)
        ),
        "random_model": bool(random_model),
        "target_dim": int(target_dim),
        "vad_labels": vad_labels,
        "class_labels": list(class_labels),
        "class_names_ja": list(class_names_ja),
        "prediction": {
            "index": prediction_index,
            "code": class_labels[prediction_index],
            "name_ja": class_names_ja[prediction_index],
        },
        "probabilities": {
            label: float(probabilities[index].item())
            for index, label in enumerate(class_labels)
        },
        "vad": {
            label: float(vad[index].item()) for index, label in enumerate(vad_labels)
        },
        "logits": {
            label: float(logits[index].item()) for index, label in enumerate(class_labels)
        },
        "linear_weights": linear_weights,
        "contributions": contributions,
        "contrast_to_runner_up": contrast,
    }


def make_contrast_payload(
    prediction_index,
    runner_up_index,
    class_labels,
    class_names_ja,
    vad_labels,
    vad,
    logits,
    probabilities,
    weight,
    bias,
):
    contribution_delta = {
        "bias": float((bias[prediction_index] - bias[runner_up_index]).item())
    }
    logit_margin_sum = contribution_delta["bias"]
    for dim_index, vad_label in enumerate(vad_labels):
        delta = (
            (weight[prediction_index, dim_index] - weight[runner_up_index, dim_index])
            * vad[dim_index]
        )
        contribution_delta[f"w_{vad_label}*{vad_label[0]}"] = float(delta.item())
        logit_margin_sum += float(delta.item())
    contribution_delta["logit_sum"] = logit_margin_sum

    return {
        "runner_up": {
            "index": int(runner_up_index),
            "code": class_labels[runner_up_index],
            "name_ja": class_names_ja[runner_up_index],
        },
        "logit_margin": float(
            (logits[prediction_index] - logits[runner_up_index]).item()
        ),
        "probability_margin": float(
            (probabilities[prediction_index] - probabilities[runner_up_index]).item()
        ),
        "contributions": contribution_delta,
    }


def _strip_prefix(state_dict, prefix):
    if any(key.startswith(prefix) for key in state_dict):
        return {
            key[len(prefix) :] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }
    return state_dict


if __name__ == "__main__":
    main()
