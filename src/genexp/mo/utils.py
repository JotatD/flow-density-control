from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from matplotlib.ticker import PercentFormatter


def calculate_reward_bin_percentages(
    scores: np.ndarray | torch.Tensor,
    reward_vectors: np.ndarray | torch.Tensor,
):
    if isinstance(scores, torch.Tensor):
        values = scores.detach().cpu().numpy()
    else:
        values = np.asarray(scores)

    if isinstance(reward_vectors, torch.Tensor):
        bins = reward_vectors.detach().cpu().numpy()
    else:
        bins = np.asarray(reward_vectors)

    matches = np.all(
        values[:, np.newaxis, :] == bins[np.newaxis, :, :],
        axis=2,
    )
    return matches.sum(axis=0) / values.shape[0] * 100.0


def plot_reward_bin_progression(
    percentages_by_epoch: np.ndarray,
    reward_vectors: np.ndarray | torch.Tensor,
    plot_path: str | Path,
    filename: str = "reward_bin_percentages.png",
):
    percentages = np.asarray(percentages_by_epoch)

    if isinstance(reward_vectors, torch.Tensor):
        bins = reward_vectors.detach().cpu().numpy()
    else:
        bins = np.asarray(reward_vectors)

    epoch_positions = np.arange(percentages.shape[0])
    labels = [
        f"({int(reward[0])}, {int(reward[1])})"
        for reward in bins
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7),
        constrained_layout=True,
    )

    if percentages.shape[0] == 1:
        bottom = 0.0
        colors = plt.get_cmap("tab10").colors
        for index, (label, percentage) in enumerate(
            zip(labels, percentages[0])
        ):
            ax.bar(
                0,
                percentage,
                bottom=bottom,
                width=0.5,
                label=label,
                color=colors[index % len(colors)],
            )
            bottom += percentage
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.stackplot(
            epoch_positions,
            percentages.T,
            labels=labels,
            alpha=0.9,
        )
        ax.set_xlim(epoch_positions[0], epoch_positions[-1])

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("Reward-bin percentages over fine-tuning")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Percentage of evaluation batch")

    tick_interval = max(1, int(np.ceil(percentages.shape[0] / 10)))
    tick_positions = list(epoch_positions[::tick_interval])
    if epoch_positions[-1] not in tick_positions:
        tick_positions.append(epoch_positions[-1])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        ["Initial" if position == 0 else str(position) for position in tick_positions]
    )
    ax.legend(
        title="Reward vector",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    output_directory = Path(plot_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_directory / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
    return ax


def plot_objective_points(
    ambient: torch.Tensor | np.ndarray,
    special: torch.Tensor | np.ndarray | None):
    if isinstance(ambient, torch.Tensor):
        ambient = ambient.detach().cpu().numpy()
    if special is not None and isinstance(special, torch.Tensor):
        special = special.detach().cpu().numpy()

    assert ambient.ndim == 2 and ambient.shape[1] == 2, f"Expected ambient objective points with shape (batch, 2); got {ambient.shape}"
    if special is not None:
        assert special.ndim == 2 and special.shape[1] == 2, f"Expected special objective points with shape (batch, 2); got {special.shape}"

    fig, ax = plt.subplots(figsize=(7, 6))
    
    ambient_x, ambient_y = ambient[:, 0], ambient[:, 1]
    ax.scatter(ambient_x, ambient_y, s=8, alpha=0.05 if special is not None else 0.25, c="gray")

    if special is not None:
        tab10_pink = plt.cm.tab10(np.linspace(0, 1, 10))[6]
        special_x = special[:, 0]
        special_y = special[:, 1]
        ax.scatter(special_x, special_y, s=18, alpha=1, c=[tab10_pink], edgecolors="none")

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
        ref_point = self.ref_point.to(device=y.device, dtype=y.dtype)

        # Sort by first objective ascending
        x0, idx = torch.sort(y[..., 0], dim=-1)
        x1 = torch.gather(y[..., 1], dim=-1, index=idx)

        # Suffix max on second objective to handle dominated points
        x1_sufmax = torch.flip(torch.cummax(torch.flip(x1, dims=[-1]), dim=-1).values, dims=[-1])

        # Points below ref_x cannot contribute. Clamping before computing widths
        # also prevents such points from inflating the first contributing rectangle.
        x0 = x0.clamp_min(ref_point[0])
        prev_x0 = torch.cat([ref_point[0].expand_as(x0[..., :1]), x0[..., :-1]], dim=-1)
        widths = (x0 - prev_x0).clamp_min(0.0)

        # Heights above ref_y
        heights = (x1_sufmax - ref_point[1]).clamp_min(0.0)

        hv = (widths * heights).sum(dim=-1)
        
        return hv.reshape(orig)

    def compute_hypervolume_botorch(self, objectives: torch.Tensor) -> torch.Tensor:
        orig = objectives.shape[:-2]
        y = objectives.reshape(
            -1,
            objectives.shape[-2],
            objectives.shape[-1],
        )
        ref_point = self.ref_point.to(device=y.device, dtype=y.dtype)
        hv_computer = Hypervolume(ref_point)
        hvs = [hv_computer.compute(batch) for batch in y]
        return objectives.new_tensor(hvs).reshape(orig)


def plot_score_density(
    scores: np.ndarray | torch.Tensor,
    plot_path: str | Path | None = None,
    filename: str | None = None,
):
    if isinstance(scores, torch.Tensor):
        values = scores.detach().cpu().numpy()
    else:
        values = np.asarray(scores)

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"scores must have shape (batch_size, 2), but received {values.shape}")

    if values.shape[0] == 0:
        raise ValueError("scores must contain at least one reward vector")

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("scores must contain numeric values")

    if not np.all(np.isfinite(values)):
        raise ValueError("scores must not contain NaN or infinite values")

    unique_rewards, counts = np.unique(
        values,
        axis=0,
        return_counts=True,
    )

    percentages = counts / values.shape[0] * 100.0

    # Matplotlib scatter sizes represent areas in points squared.
    # Keep every bubble large enough to contain its percentage label.
    minimum_bubble_size = 700.0
    bubble_scale = 2_500.0
    bubble_sizes = minimum_bubble_size + bubble_scale * (percentages / percentages.max())

    fig, ax = plt.subplots(
        figsize=(10, 7),
        constrained_layout=True,
    )

    scatter = ax.scatter(
        unique_rewards[:, 0],
        unique_rewards[:, 1],
        s=bubble_sizes,
        c=percentages,
        cmap="viridis",
        alpha=0.85,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    # Add readable percentage labels inside the bubbles.
    for reward, percentage, bubble_size in zip(
        unique_rewards,
        percentages,
        bubble_sizes,
    ):
        # Convert the marker area into an approximate diameter in points.
        bubble_diameter = 2.0 * np.sqrt(bubble_size / np.pi)

        font_size = float(
            np.clip(
                bubble_diameter * 0.28,
                7.0,
                11.0,
            )
        )

        annotation = ax.annotate(
            f"{percentage:.1f}%",
            xy=(reward[0], reward[1]),
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color="white",
            zorder=4,
        )

        # A dark outline keeps white text visible on light-colored bubbles.
        annotation.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=2.0,
                    foreground="black",
                    alpha=0.65,
                ),
                path_effects.Normal(),
            ]
        )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.02,
        shrink=0.9,
    )
    colorbar.set_label(
        "Percentage of batch",
        fontsize=11,
        labelpad=10,
    )
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    ax.set_title(
        "Reward Vector Distribution",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel(
        "Reward Objective 1",
        fontsize=12,
        labelpad=10,
    )
    ax.set_ylabel(
        "Reward Objective 2",
        fontsize=12,
        labelpad=10,
    )

    ax.grid(
        visible=True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.25,
        zorder=0,
    )
    ax.set_axisbelow(True)

    # Add padding so bubbles near the boundaries are not clipped.
    x_range = np.ptp(unique_rewards[:, 0])
    y_range = np.ptp(unique_rewards[:, 1])

    x_padding = max(x_range * 0.12, 1.0)
    y_padding = max(y_range * 0.12, 1.0)

    ax.set_xlim(
        unique_rewards[:, 0].min() - x_padding,
        unique_rewards[:, 0].max() + x_padding,
    )
    ax.set_ylim(
        unique_rewards[:, 1].min() - y_padding,
        unique_rewards[:, 1].max() + y_padding,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.5,
        1.01,
        f"Batch size: {values.shape[0]:,}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        alpha=0.7,
    )

    if plot_path is not None or filename is not None:
        output_directory = Path(plot_path) if plot_path is not None else Path(".")
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = output_directory / (filename or "score_density.png")

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(fig)
    return ax
