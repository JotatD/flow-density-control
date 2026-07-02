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
