"""Load and audit local FunASR snapshots and verify official inference parity."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .audio import sha256_file
from .contracts import LABEL_ORDER, OFFICIAL_TARGET_ORDER

OFFICIAL_EXTRACTION_VERSION = "ser_official_features_v1"
OFFICIAL_CLASSES = frozenset(("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"))
PRIMARY_NAMES = {"anger": "angry", "happy": "happy", "sadness": "sad", "disgust": "disgusted"}


def implementation_hashes() -> dict[str, str]:
    """Hash every local module that defines waveform-to-cache-to-head parity."""
    root = Path(__file__).parent
    return {name: sha256_file(root / name) for name in ("audio.py", "features.py", "model.py", "official.py")}


def implementation_sha256(hashes=None) -> str:
    values = implementation_hashes() if hashes is None else hashes
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(f"{name}:{value}\n".encode())
    return digest.hexdigest()


def normalize_official_label(value: str) -> str:
    name = str(value).strip().split("/")[-1].strip("<>").lower()
    name = "unknown" if name == "unk" else name
    if name not in OFFICIAL_CLASSES:
        raise ValueError(f"unknown official label: {value!r}")
    return name


@dataclass(frozen=True)
class OfficialLabelSpec:
    labels: tuple[str, ...]

    def __post_init__(self):
        normalized = tuple(normalize_official_label(value) for value in self.labels)
        if len(normalized) != 9 or set(normalized) != OFFICIAL_CLASSES:
            raise ValueError("official dictionary must contain nine unique classes")
        object.__setattr__(self, "labels", normalized)

    @property
    def primary_indices(self) -> tuple[int, ...]:
        return tuple(self.labels.index(PRIMARY_NAMES[label]) for label in LABEL_ORDER)

    @property
    def target_names(self) -> tuple[str, ...]:
        return OFFICIAL_TARGET_ORDER

    @property
    def target_indices(self) -> tuple[int, ...]:
        return tuple(self.labels.index(label) for label in self.target_names)

    @property
    def fixed_indices(self) -> tuple[int, ...]:
        return tuple(index for index in range(9) if index not in self.target_indices)

    def as_dict(self):
        return {
            "labels": list(self.labels),
            "primary_indices": list(self.primary_indices),
            "target_names": list(self.target_names),
            "target_indices": list(self.target_indices),
            "fixed_indices": list(self.fixed_indices),
        }


def resolve_label_spec(tokens, checkpoint_labels) -> OfficialLabelSpec:
    if isinstance(checkpoint_labels, dict):
        if any(type(index) is not int for index in checkpoint_labels.values()) or set(checkpoint_labels.values()) != set(range(9)):
            raise ValueError("checkpoint label indices must be exactly 0 through 8")
        checkpoint_labels = sorted(checkpoint_labels, key=checkpoint_labels.get)
    left, right = OfficialLabelSpec(tuple(tokens)), OfficialLabelSpec(tuple(checkpoint_labels))
    if left != right:
        raise ValueError("tokens.txt and checkpoint labels disagree")
    return left


def state_sha256(state) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}\0".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class OfficialHead:
    proj: nn.Linear
    label_spec: OfficialLabelSpec
    provenance: dict


def _read_snapshot(snapshot):
    import yaml

    root = Path(snapshot).resolve()
    config_path, checkpoint_path, tokens_path = (root / name for name in ("config.yaml", "model.pt", "tokens.txt"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model") != "Emotion2vec" or config.get("model_conf", {}).get("embed_dim") != 1024:
        raise ValueError("snapshot must describe official Emotion2vec Large")
    if type(config["model_conf"].get("normalize")) is not bool:
        raise ValueError("official normalize setting must be boolean")
    # Only the explicitly supplied local official snapshot uses legacy pickle.
    # Our generated decoder checkpoints always use weights_only=True.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("official checkpoint requires a model state dictionary")
    tokens = tokens_path.read_text(encoding="utf-8").splitlines()
    task_state = checkpoint.get("task_state")
    if not isinstance(task_state, dict) or "labels" not in task_state:
        raise ValueError("official checkpoint label dictionary is missing")
    spec = resolve_label_spec(tokens, task_state["labels"])
    state = checkpoint["model"]
    for key, shape in (("proj.weight", (9, 1024)), ("proj.bias", (9,))):
        if key not in state or tuple(state[key].shape) != shape or state[key].dtype != torch.float32:
            raise ValueError(f"official {key} must be FP32 with shape {shape}")
        if not torch.isfinite(state[key]).all():
            raise ValueError(f"non-finite official {key}")
    provenance = {
        "revision": root.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "tokens_sha256": sha256_file(tokens_path),
        "head_sha256": state_sha256({k: state[k] for k in ("proj.weight", "proj.bias")}),
        "label_spec": spec.as_dict(),
        "normalize": config["model_conf"]["normalize"],
        "mask": False, "remove_extra_tokens": True,
    }
    return config, state, spec, provenance, tokens


def load_official_head(snapshot) -> OfficialHead:
    _, state, spec, provenance, _ = _read_snapshot(snapshot)
    proj = nn.Linear(1024, 9)
    proj.load_state_dict({"weight": state["proj.weight"], "bias": state["proj.bias"]}, strict=True)
    proj.requires_grad_(False).eval()
    return OfficialHead(proj, spec, provenance)


def require_parity_report(head, report, cache_meta=None):
    """Require measured official parity for the current adapter and cache signature."""
    import json
    from .cache import validate_official_cache

    payload = json.loads(Path(report).read_text(encoding="utf-8")) if isinstance(report, (str, Path)) else report
    if (not isinstance(payload, dict) or payload.get("passed") is not True
            or payload.get("snapshot") != head.provenance
            or payload.get("rtol") != 1e-5 or payload.get("atol") != 1e-6):
        raise ValueError("a matching successful official parity report is required")
    extraction = payload.get("extraction") or {}
    hashes = implementation_hashes()
    if (payload.get("reference_api") != "funasr.AutoModel.generate"
            or extraction.get("implementation_files_sha256") != hashes
            or extraction.get("implementation_sha256") != implementation_sha256(hashes)):
        raise ValueError("official adapter changed since parity verification")
    if cache_meta is not None:
        actual = validate_official_cache(cache_meta, head.provenance)
        if actual != extraction:
            raise ValueError("cache extraction differs from verified parity environment")
    return payload


def strict_load_official_state(model, source, scope_map):
    """Resolve every persistent model tensor, rejecting missing or ambiguous keys."""
    scopes = scope_map.split(",") if isinstance(scope_map, str) else list(scope_map)
    if len(scopes) % 2:
        raise ValueError("scope_map must contain source/destination pairs")
    mapped, key_map = {}, {}
    for key, target in model.state_dict().items():
        candidates = {key} if key in source else set()
        for src, dst in zip(scopes[::2], scopes[1::2]):
            src = "" if src is None or str(src).lower() == "none" else str(src)
            dst = "" if dst is None or str(dst).lower() == "none" else str(dst)
            candidate = src + key[len(dst):] if key.startswith(dst) else None
            if candidate in source:
                candidates.add(candidate)
        if len(candidates) != 1:
            raise ValueError(f"missing or ambiguous checkpoint mapping: {key}")
        original = candidates.pop()
        tensor = source[original]
        if tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError(f"checkpoint shape/dtype mismatch: {key}")
        mapped[key], key_map[key] = tensor, original
    if len(set(key_map.values())) != len(key_map):
        raise ValueError("checkpoint mapping reuses a source tensor")
    model.load_state_dict(mapped, strict=True)
    for key, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), mapped[key].cpu()):
            raise ValueError(f"checkpoint value mismatch: {key}")
    return key_map


class OfficialEmotion2vecEncoder:
    """Frozen official final-frame encoder; normalizes each waveform exactly once."""

    def __init__(self, snapshot, *, device="auto"):
        from funasr.models.emotion2vec.model import Emotion2vec
        from .features import EncoderInfo
        from .training import resolve_device

        config, state, spec, provenance, tokens = _read_snapshot(snapshot)
        self.snapshot = Path(snapshot).resolve()
        self.config = copy.deepcopy(config)
        self.device = resolve_device(device)
        self.model = Emotion2vec(model_conf=config["model_conf"], vocab_size=9).float()
        # FunASR constructs a reconstruction decoder from the pretraining config.
        # Its official removal API drops only that decoder, unused by inference.
        self.model.modality_encoders["AUDIO"].remove_pretraining_modules(keep_decoder=False)
        self.key_map = strict_load_official_state(self.model, state, config["scope_map"])
        if not isinstance(self.model.proj, nn.Linear) or self.model.proj.weight.shape != (9, 1024) or self.model.proj.bias is None:
            raise ValueError("official proj must be Linear(1024, 9) with bias")
        self.model.requires_grad_(False).eval().to(self.device)
        self.head = OfficialHead(copy.deepcopy(self.model.proj).cpu(), spec, provenance)
        self.tokens = tokens
        hashes = implementation_hashes()
        self.provenance = {
            "snapshot": provenance,
            "extraction_code_version": OFFICIAL_EXTRACTION_VERSION,
            "implementation_sha256": implementation_sha256(hashes),
            "implementation_files_sha256": hashes,
            "remove_pretraining_decoder": True,
            "dependencies": {name: importlib.metadata.version(name) for name in ("funasr", "torch", "torchaudio", "numpy", "hydra-core", "omegaconf")},
        }
        self.info = EncoderInfo("emotion2vec_plus_large", provenance["checkpoint_sha256"], 1024)

    def extract(self, waveform):
        array = np.asarray(waveform, dtype=np.float32)
        if array.ndim != 1 or not array.size or not np.isfinite(array).all():
            raise ValueError("waveform must be a finite nonempty mono vector")
        self.model.eval()
        with torch.no_grad():
            source = torch.from_numpy(array).to(self.device)
            if self.head.provenance["normalize"]:
                source = torch.nn.functional.layer_norm(source, source.shape)
            result = self.model.extract_features(source[None], padding_mask=None, mask=False, remove_extra_tokens=True)
            x = result["x"]
            if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] == 0 or x.shape[2] != 1024:
                raise ValueError("official encoder must return [1, T, 1024]")
            if result.get("padding_mask") is not None:
                x = x[:, ~result["padding_mask"][0]]
            return x[0].cpu().numpy().copy()

    def verify_parity(self, waveform):
        """Compare official inference's hooked logits and public scores with cached C."""
        from .model import OfficialHeadModel
        from funasr import AutoModel

        started = perf_counter()
        before = state_sha256(self.model.state_dict())
        extraction_started = perf_counter()
        features = self.extract(waveform)
        extraction_seconds = perf_counter() - extraction_started
        try:
            import resource
            adapter_peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:  # pragma: no cover - Windows does not provide resource.
            adapter_peak_rss_kib = None
        c = OfficialHeadModel(self.head, condition="C").to(self.device)
        with torch.no_grad():
            logits = c(torch.from_numpy(features)[None].to(self.device))

        # Independent, unmodified official construction/loader is the reference.
        # It retains its unused reconstruction decoder; the adapter does not.
        reference_config = copy.deepcopy(self.config)
        reference_config["tokenizer_conf"]["token_list"] = self.tokens
        reference = AutoModel(
            **reference_config, init_param=str(self.snapshot / "model.pt"),
            device=str(self.device), disable_update=True, disable_pbar=True, log_level="ERROR",
            ncpu=torch.get_num_threads(),
        )
        reference.model.requires_grad_(False).eval()
        ref_state = reference.model.state_dict()
        for key, value in self.model.state_dict().items():
            if key not in ref_state or not torch.equal(value, ref_state[key]):
                raise ValueError(f"official loader and strict adapter differ: {key}")
        reference_before = state_sha256(ref_state)
        captured = []
        hook = reference.model.proj.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
        try:
            with torch.no_grad():
                results = reference.generate(
                    input=np.asarray(waveform, dtype=np.float32),
                    extract_embedding=False, fs=16000, batch_size=1,
                )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError("official inference must call proj once")
        if OfficialLabelSpec(tuple(results[0]["labels"])) != self.head.label_spec:
            raise ValueError("official generate labels differ from snapshot")
        torch.testing.assert_close(logits, captured[0], rtol=1e-5, atol=1e-6)
        scores = torch.tensor(results[0]["scores"], device=self.device)[None]
        torch.testing.assert_close(logits.softmax(-1), scores, rtol=1e-5, atol=1e-6)
        if before != state_sha256(self.model.state_dict()) or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise ValueError("official model changed during inference")
        if reference_before != state_sha256(reference.model.state_dict()):
            raise ValueError("official reference changed during inference")
        audio_seconds = len(waveform) / 16000
        try:
            import resource
            verification_peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:  # pragma: no cover - Windows does not provide resource.
            verification_peak_rss_kib = None
        return {"passed": True, "rtol": 1e-5, "atol": 1e-6, "max_logit_error": float((logits - captured[0]).abs().max()),
                "reference_api": "funasr.AutoModel.generate", "reference_retains_pretraining_decoder": True,
                "snapshot": self.head.provenance, "extraction": self.provenance, "state_sha256": before,
                "waveform_sha256": hashlib.sha256(np.asarray(waveform, dtype=np.float32).tobytes()).hexdigest(),
                "feature_shape": list(features.shape), "feature_bytes": int(features.nbytes), "device": str(self.device),
                "audio_seconds": audio_seconds, "extraction_seconds": extraction_seconds,
                "extraction_realtime_factor": extraction_seconds / audio_seconds,
                "feature_bytes_per_audio_second": features.nbytes / audio_seconds,
                "adapter_peak_rss_kib": adapter_peak_rss_kib,
                "verification_peak_rss_kib": verification_peak_rss_kib,
                "elapsed_seconds": perf_counter() - started}


__all__ = [
    "OFFICIAL_EXTRACTION_VERSION",
    "OfficialEmotion2vecEncoder",
    "OfficialHead",
    "OfficialLabelSpec",
    "load_official_head",
    "implementation_hashes",
    "implementation_sha256",
    "normalize_official_label",
    "require_parity_report",
    "resolve_label_spec",
    "state_sha256",
    "strict_load_official_state",
]
