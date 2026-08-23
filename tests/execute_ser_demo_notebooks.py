"""Execute the separated SER notebooks with long-running flags disabled."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "notebooks" / "01_extract_emotion2vec_features.ipynb",
    ROOT / "notebooks" / "02_train_and_evaluate_decoder.ipynb",
)
OUTPUT_DIR = ROOT / "runs" / "notebooks"


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        kernels = Path(directory) / "kernels" / "python3"
        kernels.mkdir(parents=True)
        (kernels / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Temporary SER demo kernel",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = directory
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for source in SOURCES:
                current = nbformat.read(source, as_version=4)
                executed = NotebookClient(current, timeout=600, kernel_name="python3").execute(cwd=ROOT)
                output = OUTPUT_DIR / f"{source.stem}.demo.executed.ipynb"
                nbformat.write(executed, output)
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
    expected = [OUTPUT_DIR / f"{source.stem}.demo.executed.ipynb" for source in SOURCES]
    if not all(path.is_file() for path in expected):
        raise RuntimeError("SER demo notebooks did not produce both executed copies")
    print("Separated SER demo notebooks completed")


if __name__ == "__main__":
    main()
