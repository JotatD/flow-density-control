import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from botorch.utils.multi_objective.hypervolume import Hypervolume

def plot_objective_points(
    ambient: torch.Tensor,
    special: torch.Tensor | None):
    ambient = ambient.detach().cpu().numpy()

    assert ambient.ndim == 2 and ambient.shape[1] == 2, f"Expected ambient objective points with shape (batch, 2); got {ambient.shape}"
    if special is not None:
        special = special.detach().cpu().numpy()
        assert special.ndim == 2 and special.shape[1] == 2, f"Expected special objective points with shape (batch, 2); got {special.shape}"

    fig, ax = plt.subplots(figsize=(7, 6))
    
    ambient_x, ambient_y = ambient[:, 0], ambient[:, 1]
    ax.scatter(ambient_x, ambient_y, s=8, alpha=0.18 if special is not None else 0.25, c="gray")

    if special is not None:
        tab10_pink = plt.cm.tab10(np.linspace(0, 1, 10))[6]
        special_x = special[:, 0]
        special_y = special[:, 1]
        ax.scatter(special_x, special_y, s=14, alpha=0.9, c=[tab10_pink], edgecolors="none")

    ax.set_xlabel("objective 1")
    ax.set_ylabel("objective 2")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.close(fig)
    return ax

def plot_clipped_values(high: float, low: float, values: np.ndarray):
    if high <= low: raise ValueError("high must be greater than low")
    values = np.asarray(values)
    x = np.arange(values.size)
    clipped = np.clip(values, low, high)
    clipped_high = values > high
    clipped_low = values < low
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(x, clipped, s=50)
    ax.scatter(x[clipped_high], np.full(clipped_high.sum(), high), marker="^", s=120, label=f"Clipped above {high}")
    ax.scatter(x[clipped_low], np.full(clipped_low.sum(), low), marker="v", s=120, label=f"Clipped below {low}")
    for i, value in enumerate(values):
        if value > high: ax.annotate(f"{value:,}", xy=(i, high), xytext=(0, 8), textcoords="offset points", ha="center")
        if value < low: ax.annotate(f"{value:,}", xy=(i, low), xytext=(0, -14), textcoords="offset points", ha="center", va="top")
    ax.axhline(0, linewidth=1)
    ax.set_ylim(low - 15, high + 15)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("Robust visualization with clipped outliers")
    plt.tight_layout()
    plt.close(fig)
    return ax

class HVComputer:
    def __init__(self, ref_point: torch.Tensor, num_rew: int = 1):
        self.ref_point = ref_point
        self.num_rew = num_rew
        assert self.ref_point.shape == (self.num_rew,), f"Expected ref_point shape ({self.num_rew},), got {self.ref_point.shape}"
        
        
    def __call__(self, objectives: torch.Tensor) -> torch.Tensor:
        if self.num_rew == 2:
            return self.compute_hypervolume_fast_2d(objectives)
        else:
            return self.compute_hypervolume_botorch(objectives)
    
    def compute_hypervolume_fast_2d(self, objectives: torch.Tensor) -> torch.Tensor:
        orig = objectives.shape[:-2]
        y = objectives.reshape(-1, objectives.shape[-2], 2)

        # Sort by first objective ascending
        x0, idx = torch.sort(y[..., 0], dim=-1)
        x1 = torch.gather(y[..., 1], dim=-1, index=idx)

        # Suffix max on second objective to handle dominated points
        x1_sufmax = torch.flip(torch.cummax(torch.flip(x1, dims=[-1]), dim=-1).values, dims=[-1])

        # Rectangle widths from ref_x and previous x
        # torch.full_like requires a Python scalar for the fill value; extract it with .item()
        prev_x0 = torch.cat([torch.full_like(x0[..., :1], self.ref_point[0].item()), x0[..., :-1]], dim=-1)
        widths = (x0 - prev_x0).clamp_min(0.0)

        # Heights above ref_y
        heights = (x1_sufmax - self.ref_point[1]).clamp_min(0.0)

        hv = (widths * heights).sum(dim=-1)
        
        return hv.reshape(orig)

    def compute_hypervolume_botorch(self, objectives: torch.Tensor) -> torch.Tensor:
        hv_computer = Hypervolume(self.ref_point)
        og_shape = objectives.shape
        if len(objectives.shape) > 3:
            k_ = objectives.shape[-1]
            n_ = objectives.shape[-2] 
            objectives = objectives.reshape(-1, n_, k_)
        if len(objectives.shape) == 3:
            hvs = [hv_computer.compute(rew) for rew in objectives]
        elif len(objectives.shape) == 2:
            hvs = [hv_computer.compute(objectives)]
        return torch.tensor(hvs, device=objectives.device).reshape(og_shape[:-2])