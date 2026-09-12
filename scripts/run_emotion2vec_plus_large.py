"""Run the public emotion2vec+ large classifier on one audio file via FunASR."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL = "iic/emotion2vec_plus_large"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the public emotion2vec+ large 9-class emotion classifier."
    )
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"FunASR model ID or local model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--hub",
        choices=("ms", "modelscope", "hf", "huggingface"),
        default="hf",
        help="Model download service (default: hf)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device, for example cpu or cuda:0 (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/emotion2vec_plus_large"),
        help="Directory used by FunASR for outputs",
    )
    parser.add_argument(
        "--extract-embedding",
        action="store_true",
        help="Also request the emotion representation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio = args.audio.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from funasr import AutoModel

    model = AutoModel(model=args.model, hub=args.hub, device=args.device)
    results = model.generate(
        input=str(audio),
        output_dir=str(output_dir),
        granularity="utterance",
        extract_embedding=args.extract_embedding,
    )
    if not results or "labels" not in results[0] or "scores" not in results[0]:
        raise RuntimeError("FunASR did not return labels and scores")

    labels = list(results[0]["labels"])
    scores = [float(score) for score in results[0]["scores"]]
    if len(labels) != len(scores):
        raise RuntimeError("FunASR returned different label and score counts")

    print(f"audio: {audio}")
    print(f"model: {args.model}")
    print("scores:")
    for label, score in zip(labels, scores):
        print(f"  {label}: {score:.6f}")
    if scores:
        best_index = max(range(len(scores)), key=scores.__getitem__)
        print(f"top emotion: {labels[best_index]} ({scores[best_index]:.6f})")

    if args.extract_embedding and "feats" in results[0]:
        shape = getattr(results[0]["feats"], "shape", None)
        print(f"embedding shape: {shape if shape is not None else 'available'}")
    print(f"output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
