import time
from collections import defaultdict
from contextlib import contextmanager

import torch
from diffusiongym.base_models import BaseModel
from diffusiongym.types import D


def _score_func(model: BaseModel[D], latent: D, t: torch.Tensor) -> D:
    """Compute the score function ∇log p_t(x) from a general model output."""
    if model.output_type == "score":
        return model.forward(latent, t)

    if model.output_type == "velocity":
        v = model.forward(latent, t)
        scheduler = model.scheduler
        kappa = scheduler.kappa(latent, t)
        eta = scheduler.eta(latent, t)
        return (v - kappa * latent) / eta

    if model.output_type == "endpoint":
        x_1 = model.forward(latent, t)
        scheduler = model.scheduler
        alpha = scheduler.alpha(latent, t)
        beta = scheduler.beta(latent, t)
        return (alpha * x_1 - latent) / (beta**2)

    if model.output_type == "epsilon":
        eps = model.forward(latent, t)
        beta = model.scheduler.beta(latent, t)
        return -eps / beta

    raise ValueError("Incorrectly specified base model")


class StepTimer:
    def __init__(self, device=None):
        self.times = defaultdict(list)
        self.device = device
        self._use_cuda_sync = (isinstance(device, torch.device) and device.type == "cuda") or (
            isinstance(device, str) and "cuda" in device
        )

    @contextmanager
    def section(self, name):
        if self._use_cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._use_cuda_sync:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print("Name inserted: ", name, "Time taken: ", dt)
            self.times[name].append(dt)

    def summary(self, top_k=None):
        # returns (name, count, total, mean, p50, p95)
        import numpy as np

        rows = []
        for k, v in self.times.items():
            a = np.array(v, dtype=float)
            rows.append((k, len(a), a.sum(), a.mean(), np.median(a), np.percentile(a, 95)))
        rows.sort(key=lambda r: r[2], reverse=True)  # by total time
        return rows[:top_k] if top_k else rows