import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from botorch.utils.multi_objective.hypervolume import Hypervolume

def plot_objective_points(
    ambient: torch.Tensor,
    special: torch.Tensor | None,
    save_path: Path,
):
    special = special.detach().cpu().numpy()
    ambient = ambient.detach().cpu().numpy()

    assert ambient.ndim == 2 and ambient.shape[1] == 2, f"Expected ambient objective points with shape (batch, 2); got {ambient.shape}"
    if special is not None:
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

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved objective-point figure: {save_path}")


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