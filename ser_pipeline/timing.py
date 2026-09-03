"""Wall-clock stage and data-loader timing shared by training and evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import torch


@contextmanager
def measure(timings, key, device=None):
    """Accumulate elapsed seconds, synchronizing CUDA only when timing is enabled."""
    if timings is None:
        yield
        return
    cuda_device = torch.device(device) if device is not None else None
    if cuda_device is not None and cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
    started = perf_counter()
    try:
        yield
    finally:
        if cuda_device is not None and cuda_device.type == "cuda":
            torch.cuda.synchronize(cuda_device)
        timings[key] = timings.get(key, 0.0) + perf_counter() - started


def timed_batches(loader, timings):
    """Measure iterator startup and batch preparation separately from computation."""
    with measure(timings, "batch_prepare_seconds"):
        iterator = iter(loader)
    while True:
        with measure(timings, "batch_prepare_seconds"):
            try:
                batch = next(iterator)
            except StopIteration:
                return
        yield batch
