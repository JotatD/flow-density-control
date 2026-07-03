import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

def plot_objective_points(
    ambient: torch.Tensor,
    special: torch.Tensor | None,
    save_path: Path,
):
    if isinstance(special, torch.Tensor):
        special = special.detach().cpu().numpy()
    if isinstance(ambient, torch.Tensor):
        ambient = ambient.detach().cpu().numpy()

    assert ambient.ndim == 2 and ambient.shape[1] == 2, f"Expected ambient objective points with shape (n, 2); got {ambient.shape}"
    if special is not None:
        assert special.ndim == 3 and special.shape[2] == 2, f"Expected special objective points with shape (batch, n_points, 2); got {special.shape}"

    fig, ax = plt.subplots(figsize=(7, 6))
    
    ambient_x, ambient_y = ambient[:, 0], ambient[:, 1]
    ax.scatter(ambient_x, ambient_y, s=8, alpha=0.18 if special is not None else 0.25, c="gray")

    if special is not None:
        n_points = special.shape[1]
        point_colors = plt.cm.tab10(np.linspace(0, 1, max(1, n_points)))
        for point_idx in range(n_points):
            special_x = special[:, point_idx, 0]
            special_y = special[:, point_idx, 1]
            ax.scatter(special_x, special_y, s=14, alpha=0.9, c=[point_colors[point_idx]], edgecolors="none")

    ax.set_xlabel("objective 1")
    ax.set_ylabel("objective 2")
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved objective-point figure: {save_path}")

    