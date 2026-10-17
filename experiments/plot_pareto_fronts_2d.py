import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flow-density-control-matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_DIFFUSION_NFT_RUN = [
    "hv_nft_20260904_231008_912634",
    "hv_nft_20260905_054947_079203",
    "hv_nft_20260905_055005_607598",
][0]
DEFAULT_DIFFUSION_BLEND_RUN = [
    "diffusion_blend_20260910_153136_855622",
    "diffusion_blend_20260910_170427_433467",
    "diffusion_blend_20260910_171858_327955",
][2]
DEFAULT_HYPERMOG_RUN = [
    "hv_nft_20260904_231010_785991",
    "hv_nft_20260905_054949_928698",
    "hv_nft_20260905_055004_365055",
][1]

DEFAULT_BASE_MODEL_RUN = [
    "geom_base_model_20260910_173819_303827",
    "geom_base_model_20260910_173819_354273",
    "geom_base_model_20260910_174158_064049",
][2]

PLOT_STYLES = {
    "diffusion": {"color": "#2563EB", "marker": "o"},
    "hypermog": {"color": "#E85D04", "marker": "D"},
}


@dataclass
class ParetoFront:
    run: str
    method: str
    side: str
    all_values: np.ndarray
    pareto_mask: np.ndarray
    pareto_values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the DiffusionNFT, DiffusionBlend, and base-model Pareto fronts against HyperMOG."
    )
    parser.add_argument("csv_path", type=Path, help="retro_compatibility_summary.csv path")
    parser.add_argument("--diffusion-nft-run", default=DEFAULT_DIFFUSION_NFT_RUN)
    parser.add_argument("--diffusion-blend-run", default=DEFAULT_DIFFUSION_BLEND_RUN)
    parser.add_argument("--base-model-run", default=DEFAULT_BASE_MODEL_RUN)
    parser.add_argument("--hypermog-run", default=DEFAULT_HYPERMOG_RUN)
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: beside the CSV)")
    return parser.parse_args()


def resolve_recorded_path(csv_path: Path, recorded_path: str) -> Path:
    path = Path(recorded_path).expanduser()
    if not path.is_absolute():
        path = csv_path.parent / path
    return path.resolve()


def load_pareto_indices(row: dict[str, str]) -> list[int]:
    try:
        indices = json.loads(row["pareto_indices"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid pareto_indices for run {row.get('run', '<unknown>')}") from exc
    if (
        not isinstance(indices, list)
        or not indices
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError(f"Expected unique integer Pareto indices for {row.get('run', '<unknown>')}")
    return indices


def load_pareto_front(
    csv_path: Path, row: dict[str, str], method: str, side: str
) -> ParetoFront:
    try:
        epoch_value = float(row["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid epoch for run {row.get('run', '<unknown>')}") from exc
    if not math.isfinite(epoch_value) or not epoch_value.is_integer():
        raise ValueError(f"Invalid epoch for run {row.get('run', '<unknown>')}: {epoch_value!r}")

    epoch = int(epoch_value)
    indices = load_pareto_indices(row)
    run_dir = resolve_recorded_path(csv_path, row["objs"]).parent
    score_path = run_dir / "scores" / f"score_{epoch}.npz"
    if not score_path.is_file():
        raise ValueError(f"Missing score archive for {row['run']}: {score_path}")

    with np.load(score_path, allow_pickle=False) as archive:
        if "scores" not in archive or "validity" not in archive:
            raise ValueError(f"Expected scores and validity arrays in {score_path}")
        scores = np.asarray(archive["scores"])
        validity = np.asarray(archive["validity"])
    if scores.ndim != 2 or scores.shape[1] != 2 or validity.shape != (len(scores),):
        raise ValueError(f"Invalid score archive shapes for {row['run']}")
    if any(index < 0 or index >= len(scores) for index in indices):
        raise ValueError(f"Pareto index outside the score range for {row['run']}")
    if not np.all(validity[indices]):
        raise ValueError(f"Pareto indices include an invalid molecule for {row['run']}")

    valid_indices = np.flatnonzero(validity)
    all_values = np.asarray(scores[validity], dtype=float)
    pareto_mask = np.isin(valid_indices, indices)
    pareto_values = np.asarray(scores[indices], dtype=float)
    if not np.all(np.isfinite(all_values)) or not np.all(np.isfinite(pareto_values)):
        raise ValueError(f"Pareto front contains non-finite values for {row['run']}")
    return ParetoFront(
        run=row["run"],
        method=method,
        side=side,
        all_values=all_values,
        pareto_mask=pareto_mask,
        pareto_values=pareto_values,
    )


def load_requested_fronts(csv_path: Path, requested: dict[str, tuple[str, str]]) -> dict[str, ParetoFront]:
    try:
        with csv_path.open(newline="") as csv_file:
            rows_by_run = {row["run"]: row for row in csv.DictReader(csv_file)}
    except OSError as exc:
        raise ValueError(f"Could not read representative CSV {csv_path}: {exc}") from exc

    missing = [run for run in requested if run not in rows_by_run]
    if missing:
        raise ValueError(f"Missing requested run(s) in {csv_path}: {', '.join(missing)}")
    return {
        run: load_pareto_front(csv_path, rows_by_run[run], method, side)
        for run, (method, side) in requested.items()
    }


def dominates_any(values: np.ndarray, comparison_front: np.ndarray) -> np.ndarray:
    at_least_as_good = values[:, None, :] >= comparison_front[None, :, :]
    strictly_better = values[:, None, :] > comparison_front[None, :, :]
    return np.any(np.all(at_least_as_good, axis=2) & np.any(strictly_better, axis=2), axis=1)


def unique_front_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_values, multiplicities = np.unique(values, axis=0, return_counts=True)
    order = np.argsort(unique_values[:, 0], kind="stable")
    return unique_values[order], multiplicities[order]


def plot_comparison(diffusion_front: ParetoFront, hypermog_front: ParetoFront, output_path: Path) -> None:
    fig, objective_ax = plt.subplots(figsize=(8.4, 6.4), facecolor="#FFFFFF", constrained_layout=True)
    objective_ax.set_facecolor("#FAFBFD")
    legend_handles = []

    for front, other_front in ((diffusion_front, hypermog_front), (hypermog_front, diffusion_front)):
        style = PLOT_STYLES[front.side]
        cross_dominating_mask = dominates_any(front.all_values, other_front.pareto_values) & ~front.pareto_mask
        cross_dominating_values = front.all_values[cross_dominating_mask]
        objective_ax.scatter(
            cross_dominating_values[:, 0],
            cross_dominating_values[:, 1],
            color=style["color"],
            marker="o",
            s=48,
            alpha=0.25,
            edgecolors="#FFFFFF",
            linewidths=0.35,
            zorder=1,
        )

        unique_values, multiplicities = unique_front_values(front.pareto_values)
        objective_ax.plot(
            unique_values[:, 0],
            unique_values[:, 1],
            color=style["color"],
            linewidth=2.2,
            alpha=0.72,
            drawstyle="steps-post",
            zorder=2,
        )
        marker_sizes = np.clip(150 + 28 * np.sqrt(multiplicities - 1), 150, 300)
        objective_ax.scatter(
            unique_values[:, 0],
            unique_values[:, 1],
            color=style["color"],
            marker=style["marker"],
            s=marker_sizes,
            edgecolors="#FFFFFF",
            linewidths=1.35,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                markersize=9,
                markeredgecolor="#FFFFFF",
                markeredgewidth=1,
                linewidth=2.2,
                label=f"{front.method}  ({len(front.pareto_values)} samples, {len(unique_values)} locations)",
            )
        )

    all_values = np.concatenate((diffusion_front.pareto_values, hypermog_front.pareto_values), axis=0)
    spans = np.maximum(np.ptp(all_values, axis=0), 0.05)
    padding = spans * 0.045
    objective_ax.set_xlim(all_values[:, 0].min() - padding[0], all_values[:, 0].max() + padding[0])
    objective_ax.set_ylim(all_values[:, 1].min() - padding[1], all_values[:, 1].max() + padding[1])
    objective_ax.set_title(
        f"{diffusion_front.method} vs {hypermog_front.method}",
        loc="left",
        fontsize=19,
        fontweight="bold",
        color="#0F172A",
        pad=14,
    )
    objective_ax.set_xlabel("QED  →", fontsize=12.5, fontweight="semibold", color="#334155", labelpad=9)
    objective_ax.set_ylabel("Normalized SA score  →", fontsize=12.5, fontweight="semibold", color="#334155", labelpad=9)
    objective_ax.text(
        0,
        1.012,
        "Higher is better for both objectives",
        transform=objective_ax.transAxes,
        color="#64748B",
        fontsize=9.5,
        va="bottom",
    )
    objective_ax.text(
        1,
        1.012,
        "Marker size reflects repeated score pairs",
        transform=objective_ax.transAxes,
        color="#64748B",
        fontsize=9.5,
        ha="right",
        va="bottom",
    )
    objective_ax.set_axisbelow(True)
    objective_ax.grid(True, color="#CBD5E1", linestyle=(0, (3, 4)), linewidth=0.8, alpha=0.65)
    objective_ax.spines["top"].set_visible(False)
    objective_ax.spines["right"].set_visible(False)
    for spine in (objective_ax.spines["left"], objective_ax.spines["bottom"]):
        spine.set_color("#94A3B8")
        spine.set_linewidth(1.05)
    objective_ax.tick_params(axis="both", colors="#475569", labelsize=10.5, length=4, width=0.9)
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#64748B",
            marker="o",
            markersize=6,
            markerfacecolor="#64748B",
            markeredgecolor="none",
            linewidth=0,
            alpha=0.45,
            label="Cross-dominating non-front sample",
        )
    )
    objective_ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        framealpha=0.96,
        fontsize=9.5,
        handlelength=2.5,
        borderpad=0.8,
        labelspacing=0.65,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    config = parse_args()
    csv_path = config.csv_path.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve() if config.output_dir is not None else csv_path.parent
    requested = {
        config.diffusion_nft_run: ("DiffusionNFT", "diffusion"),
        config.diffusion_blend_run: ("DiffusionBlend", "diffusion"),
        config.base_model_run: ("Base Model", "diffusion"),
        config.hypermog_run: ("HyperMOG", "hypermog"),
    }
    fronts = load_requested_fronts(csv_path, requested)

    comparisons = (
        (fronts[config.diffusion_nft_run], "pareto_front_diffusion_nft_vs_hypermog.png"),
        (fronts[config.diffusion_blend_run], "pareto_front_diffusion_blend_vs_hypermog.png"),
        (fronts[config.base_model_run], "pareto_front_base_model_vs_hypermog.png"),
    )
    for diffusion_front, filename in comparisons:
        output_path = output_dir / filename
        plot_comparison(diffusion_front, fronts[config.hypermog_run], output_path)
        print(f"Saved {diffusion_front.method} Pareto-front comparison to {output_path}")


if __name__ == "__main__":
    main()
