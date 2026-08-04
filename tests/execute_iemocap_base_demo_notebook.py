"""Execute and privacy-check the IEMOCAP Base demo notebook."""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "iemocap_base_downstream_training.ipynb"
OUTPUT = ROOT / "runs" / "notebooks" / "iemocap_base_downstream_training.demo.executed.ipynb"


def main():
    with tempfile.TemporaryDirectory() as directory:
        kernels = Path(directory) / "kernels" / "python3"
        kernels.mkdir(parents=True)
        (kernels / "kernel.json").write_text(
            json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Temporary IEMOCAP demo kernel", "language": "python"}),
            encoding="utf-8",
        )
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = directory
        os.environ["IEMOCAP_NOTEBOOK_MODE"] = "demo"
        try:
            notebook = nbformat.read(SOURCE, as_version=4)
            executed = NotebookClient(notebook, timeout=600, kernel_name="python3").execute(cwd=ROOT)
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
        output_text = json.dumps([cell.get("outputs", []) for cell in executed.cells], ensure_ascii=False)
        if re.search(r"Ses0[1-5][FM]?_", output_text):
            raise RuntimeError("an utterance identifier appeared in notebook output")
        private_values = [os.environ.get(name, "") for name in ("IEMOCAP_ROOT", "IEMOCAP_WORK_DIR", "EMOTION2VEC_CHECKPOINT", "EMOTION2VEC_USER_DIR")]
        if any(value and value in output_text for value in private_values):
            raise RuntimeError("a private data path appeared in notebook output")
        for required_label in ('base', 'trial', 'train', 'validation', 'test', 'macro_f1'):
            if required_label not in output_text:
                raise RuntimeError(f"the notebook output is missing {required_label!r}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(executed, OUTPUT)
    expected = [
        ROOT / "runs" / "iemocap_base_downstream" / "base_history.png",
        ROOT / "runs" / "iemocap_base_downstream" / "trial_history.png",
    ]
    if not all(path.is_file() for path in expected):
        raise RuntimeError("the demo notebook did not create all aggregate artifacts")
    if len(list((ROOT / "runs" / "iemocap_base_downstream").glob("validation_*.pt"))) < 2:
        raise RuntimeError("the demo notebook did not save both validation checkpoints")
    print("IEMOCAP Base demo notebook completed with privacy-safe outputs")


if __name__ == "__main__":
    main()
