"""Execute the CPU demo notebook in the current isolated Python environment."""

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "audio_to_emotion_vad.ipynb"
OUTPUT = ROOT / "runs" / "notebooks" / "audio_to_emotion_vad.demo.executed.ipynb"


def main():
    with tempfile.TemporaryDirectory() as directory:
        kernels = Path(directory) / "kernels" / "python3"
        kernels.mkdir(parents=True)
        (kernels / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Temporary emotion2vec demo kernel",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        os.environ["JUPYTER_PATH"] = directory
        notebook = nbformat.read(SOURCE, as_version=4)
        executed = NotebookClient(notebook, timeout=600, kernel_name="python3").execute(cwd=ROOT)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(executed, OUTPUT)
    required = [
        "audio_to_emotion_vad/model.pt",
        "audio_to_emotion_vad/emotion_labels.json",
        "audio_to_emotion_vad/training_history.json",
        "audio_to_emotion_vad/test_metrics.json",
        "audio_to_emotion_vad/evaluation_graphs.png",
        "audio_to_emotion_vad/inference_results.csv",
    ]
    missing = [name for name in required if not (ROOT / "runs" / "notebooks" / name).is_file()]
    if missing:
        raise RuntimeError(f"notebook did not create expected artifacts: {missing}")
    print(f"Demo notebook completed: {OUTPUT}")


if __name__ == "__main__":
    main()
