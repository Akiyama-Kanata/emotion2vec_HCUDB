"""Compatibility re-export of the dataset-independent SER decoder."""

try:
    from ser_pipeline.model import BaseModel
except ModuleNotFoundError:  # legacy `cd iemocap_downstream && python main.py`
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from ser_pipeline.model import BaseModel

__all__ = ["BaseModel"]
