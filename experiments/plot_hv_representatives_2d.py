import argparse
import csv
import json
import math
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flow-density-control-matplotlib"))

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageOps

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "Roboto", "font.sans-serif": ["Roboto", "Liberation Sans", "DejaVu Sans"]})

from genexp.mo.utils import HVComputer

from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

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

COLORS = {"diffusion": "#FC8F54", "hypermog": "#62109F"}
THUMBNAIL_POSITIONS = {
    "diffusion": (0.025, (0.72, 0.505, 0.29, 0.075)),
    "hypermog": (0.873, (0.72, 0.505, 0.29, 0.075)),
}
THUMBNAIL_HEIGHT = 0.18
SAMPLE_SIZE = 100
NUM_REPRESENTATIVES = 4


@dataclass
class RunRepresentatives:
    run: str
    method: str
    side: str
    epoch: int
    values: np.ndarray
    representative_indices: list[int]
    object_path: Path
    renders_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the selected DiffusionNFT and DiffusionBlend representatives against HVFT."
    )
    parser.add_argument("csv_path", type=Path, help="retro_compatibility_summary.csv path")
    parser.add_argument("--diffusion-nft-run", default=DEFAULT_DIFFUSION_NFT_RUN)
    parser.add_argument("--diffusion-blend-run", default=DEFAULT_DIFFUSION_BLEND_RUN)
    parser.add_argument("--hypermog-run", default=DEFAULT_HYPERMOG_RUN)
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: beside the CSV)")
    parser.add_argument("--sample-seed", type=int, default=0, help="Random seed for reproducible 100-point sampling")
    parser.add_argument("--force-render", action="store_true", help="Regenerate individual molecule PNGs")
    return parser.parse_args()


def resolve_recorded_path(csv_path: Path, recorded_path: str) -> Path:
    path = Path(recorded_path).expanduser()
    if not path.is_absolute():
        path = csv_path.parent / path
    return path.resolve()


def load_invalid_value(run_dir: Path, run_name: str) -> float:
    manifest_path = run_dir / "run_manifest.json"
    try:
        with manifest_path.open() as manifest_file:
            manifest = json.load(manifest_file)
        configuration = manifest["configuration"]
        invalid_value = float(configuration.get("invalid_val", -1.0))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load invalid_val for {run_name} from {manifest_path}") from exc
    if not math.isfinite(invalid_value):
        raise ValueError(f"Non-finite invalid_val for {run_name}: {invalid_value!r}")
    return invalid_value


def greedy_hypervolume_indices(scores: np.ndarray, invalid_value: float) -> tuple[list[int], float]:
    if len(scores) < NUM_REPRESENTATIVES:
        raise ValueError(f"Cannot select {NUM_REPRESENTATIVES} representatives from {len(scores)} scores")

    rewards = torch.as_tensor(scores, dtype=torch.float32)
    hv_computer = HVComputer(
        ref_point=torch.full((scores.shape[1],), invalid_value, dtype=torch.float32),
        num_rew=scores.shape[1],
    )
    selected: list[int] = []
    remaining = list(range(len(rewards)))
    selected_hypervolume = 0.0

    for _ in range(NUM_REPRESENTATIVES):
        best_index = remaining[0]
        best_hypervolume = float("-inf")
        for candidate_index in remaining:
            candidate_subset = selected + [candidate_index]
            candidate_hypervolume = float(hv_computer(rewards[candidate_subset]).detach().cpu().item())
            if candidate_hypervolume > best_hypervolume:
                best_index = candidate_index
                best_hypervolume = candidate_hypervolume
        selected.append(best_index)
        remaining.remove(best_index)
        selected_hypervolume = best_hypervolume

    return selected, selected_hypervolume


def load_representative_run(
    csv_path: Path, row: dict[str, str], method: str, side: str, rng: np.random.Generator
) -> RunRepresentatives:
    try:
        epoch_value = float(row["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid epoch for run {row.get('run', '<unknown>')}") from exc
    if not math.isfinite(epoch_value) or not epoch_value.is_integer():
        raise ValueError(f"Invalid epoch for run {row.get('run', '<unknown>')}: {epoch_value!r}")

    epoch = int(epoch_value)
    objs_dir = resolve_recorded_path(csv_path, row["objs"])
    run_dir = objs_dir.parent
    object_path = objs_dir / f"obj_{epoch}.pkl"
    score_path = run_dir / "scores" / f"score_{epoch}.npz"
    if not object_path.is_file():
        raise ValueError(f"Missing molecule-object file for {row['run']}: {object_path}")
    if not score_path.is_file():
        raise ValueError(f"Missing score archive for {row['run']}: {score_path}")

    with np.load(score_path, allow_pickle=False) as archive:
        if "scores" not in archive or "validity" not in archive:
            raise ValueError(f"Expected scores and validity arrays in {score_path}")
        scores = np.asarray(archive["scores"])
        validity = np.asarray(archive["validity"])
    if scores.ndim != 2 or scores.shape[1] != 2 or validity.shape != (len(scores),):
        raise ValueError(f"Invalid score archive shapes for {row['run']}")
    if validity.dtype != np.bool_:
        raise ValueError(f"Expected a boolean validity array for {row['run']}")

    valid_indices = np.flatnonzero(validity)
    if len(valid_indices) < SAMPLE_SIZE:
        raise ValueError(f"Cannot sample {SAMPLE_SIZE} valid molecules for {row['run']}; only {len(valid_indices)} available")
    valid_scores = np.asarray(scores[valid_indices], dtype=float)
    if not np.all(np.isfinite(valid_scores)):
        raise ValueError(f"Valid scores include non-finite values for {row['run']}")

    sampled_positions = rng.choice(len(valid_indices), size=SAMPLE_SIZE, replace=False)
    sampled_indices = valid_indices[sampled_positions]
    sampled_scores = valid_scores[sampled_positions]
    selected_positions, _ = greedy_hypervolume_indices(sampled_scores, load_invalid_value(run_dir, row["run"]))
    indices = [int(sampled_indices[position]) for position in selected_positions]
    values = np.asarray(sampled_scores[selected_positions], dtype=float)
    return RunRepresentatives(
        run=row["run"],
        method=method,
        side=side,
        epoch=epoch,
        values=values,
        representative_indices=indices,
        object_path=object_path,
        renders_dir=run_dir / "renders",
    )


def load_requested_runs(
    csv_path: Path, requested: dict[str, tuple[str, str]], sample_seed: int
) -> dict[str, RunRepresentatives]:
    try:
        with csv_path.open(newline="") as csv_file:
            rows_by_run = {row["run"]: row for row in csv.DictReader(csv_file)}
    except OSError as exc:
        raise ValueError(f"Could not read representative CSV {csv_path}: {exc}") from exc

    missing = [run for run in requested if run not in rows_by_run]
    if missing:
        raise ValueError(f"Missing requested run(s) in {csv_path}: {', '.join(missing)}")
    rng = np.random.default_rng(sample_seed)
    return {
        run: load_representative_run(csv_path, rows_by_run[run], method, side, rng)
        for run, (method, side) in requested.items()
    }


def render_thumbnails(run: RunRepresentatives, force_render: bool) -> list[Path]:
    run.renders_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_paths = [run.renders_dir / f"render_{index}.png" for index in run.representative_indices]
    missing = [index for index, path in enumerate(thumbnail_paths) if force_render or not path.is_file()]
    if not missing:
        return thumbnail_paths

    from render_molecule_3d import render_molecule_3d

    try:
        with run.object_path.open("rb") as molecule_file:
            molecules = pickle.load(molecule_file)
    except Exception as exc:
        raise ValueError(f"Could not load molecules from {run.object_path}: {exc}") from exc
    if not isinstance(molecules, list):
        raise ValueError(f"Expected {run.object_path} to contain a molecule list")

    for representative_position in missing:
        molecule_index = run.representative_indices[representative_position]
        if molecule_index >= len(molecules):
            raise ValueError(f"Representative index {molecule_index} is outside {run.object_path}")
        render_molecule_3d(molecules[molecule_index], thumbnail_paths[representative_position])
    return thumbnail_paths


def crop_white_background(image_path: Path, margin: int = 10) -> Image.Image:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        pixels = np.asarray(rgb_image)
        foreground = np.any(pixels < 248, axis=2)
        if not foreground.any():
            raise ValueError(f"Rendered molecule image is blank: {image_path}")
        rows, columns = np.nonzero(foreground)
        left = max(0, int(columns.min()) - margin)
        upper = max(0, int(rows.min()) - margin)
        right = min(rgb_image.width, int(columns.max()) + margin + 1)
        lower = min(rgb_image.height, int(rows.max()) + margin + 1)
        return rgb_image.crop((left, upper, right, lower))


def square_thumbnail(image_path: Path, size: int = 512, padding: int = 10) -> Image.Image:
    cropped = crop_white_background(image_path)
    molecule = ImageOps.contain(cropped, (size - 2 * padding, size - 2 * padding), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(molecule, ((size - molecule.width) // 2, (size - molecule.height) // 2))
    return canvas


def add_thumbnail_column(
    fig: plt.Figure,
    objective_ax: plt.Axes,
    run: RunRepresentatives,
    thumbnail_paths: list[Path],
) -> None:
    color = COLORS[run.side]
    x_position, y_positions = THUMBNAIL_POSITIONS[run.side]
    side_is_left = run.side == "diffusion"
    target_x = 1.0 if side_is_left else 0.0
    thumbnail_width = THUMBNAIL_HEIGHT * fig.get_figheight() / fig.get_figwidth()

    for y_position, representative_index in zip(y_positions, np.argsort(run.values[:, 1])[::-1]):
        thumbnail_ax = fig.add_axes((x_position, y_position, thumbnail_width, THUMBNAIL_HEIGHT), facecolor="white")
        thumbnail_ax.imshow(square_thumbnail(thumbnail_paths[int(representative_index)]))
        thumbnail_ax.set_xticks([])
        thumbnail_ax.set_yticks([])
        for spine in thumbnail_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(1.35)
            spine.set_linestyle("-")
            spine.set_alpha(0.85)

        qed, sa = run.values[representative_index]
        thumbnail_ax.set_title(
            f"QED {qed:.3f}  ·  SA {sa:.3f}",
            color=color,
            fontsize=12.5,
            fontweight="medium",
            pad=7,
            alpha=1.0,
        )
        connection = ConnectionPatch(
            xyA=(float(qed), float(sa)),
            coordsA=objective_ax.transData,
            xyB=(target_x, 0.5),
            coordsB=thumbnail_ax.transAxes,
            color=color,
            linewidth=1.0 if run.side == "hypermog" else 0.75,
            linestyle="-" if run.side == "hypermog" else (0, (8, 5)),
            alpha=0.38 if run.side == "hypermog" else 0.30,
            arrowstyle="-",
            connectionstyle=f"arc3,rad={-0.04 if side_is_left else 0.04}",
            clip_on=False,
            zorder=1,
        )
        fig.add_artist(connection)


def plot_comparison(
    diffusion_run: RunRepresentatives,
    hypermog_run: RunRepresentatives,
    thumbnails: dict[str, list[Path]],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(15, 8.5), facecolor="white")
    objective_ax = fig.add_axes((0.22, 0.09, 0.56, 0.84), facecolor="white")

    runs = (diffusion_run, hypermog_run)
    all_values = np.concatenate([run.values for run in runs], axis=0)
    spans = np.maximum(np.ptp(all_values, axis=0), 0.05)
    padding = spans * 0.035
    objective_ax.set_xlim(all_values[:, 0].min() - padding[0], all_values[:, 0].max() + padding[0])
    objective_ax.set_ylim(all_values[:, 1].min() - padding[1], all_values[:, 1].max() + padding[1])

    for run in runs:
        color = COLORS[run.side]
        order = np.argsort(run.values[:, 0])
        points = run.values[order]
        objective_ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=2.8,
            linestyle=(0, (9, 5)) if run.side == "diffusion" else "-",
            alpha=0.72,
            drawstyle="steps-post",
            zorder=2,
        )
        objective_ax.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            marker="D" if run.side == "hypermog" else "o",
            s=255,
            edgecolors="#FFFFFF",
            linewidths=1.9,
            zorder=4,
        )
        add_thumbnail_column(fig, objective_ax, run, thumbnails[run.run])

    fig.text(0.08, 0.965, diffusion_run.method, ha="center", va="top", color=COLORS["diffusion"], fontsize=20, fontweight="medium")
    fig.text(0.925, 0.965, hypermog_run.method, ha="center", va="top", color=COLORS["hypermog"], fontsize=20, fontweight="medium")
    objective_ax.set_xlabel("QED", fontsize=17, fontweight="medium", color="#334155", labelpad=10)
    objective_ax.set_ylabel("Normalized SA score", fontsize=17, fontweight="medium", color="#334155", labelpad=11)
    objective_ax.yaxis.set_label_coords(-0.06, 0.5)
    objective_ax.set_axisbelow(True)
    objective_ax.grid(True, color="#CBD5E1", linestyle=(0, (1, 4)), linewidth=0.85, alpha=0.58)
    objective_ax.spines["top"].set_visible(False)
    objective_ax.spines["right"].set_visible(False)
    axis_color = "#334155"
    for spine in (objective_ax.spines["left"], objective_ax.spines["bottom"]):
        spine.set_color(axis_color)
        spine.set_linewidth(1.8)
    objective_ax.minorticks_on()
    objective_ax.tick_params(axis="both", which="major", direction="out", colors=axis_color, labelsize=13.5, length=7, width=1.5)
    objective_ax.tick_params(axis="both", which="minor", direction="out", colors=axis_color, length=3.5, width=1.0)
    arrow_style = {"arrowstyle": "-|>", "color": axis_color, "linewidth": 1.8, "mutation_scale": 12}
    objective_ax.annotate("", xy=(1.012, 0), xytext=(0.985, 0), xycoords="axes fraction", arrowprops=arrow_style, annotation_clip=False)
    objective_ax.annotate("", xy=(0, 1.012), xytext=(0, 0.985), xycoords="axes fraction", arrowprops=arrow_style, annotation_clip=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    config = parse_args()
    csv_path = config.csv_path.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve() if config.output_dir is not None else csv_path.parent
    requested = {
        config.diffusion_nft_run: ("DiffusionNFT", "diffusion"),
        config.diffusion_blend_run: ("DiffusionBlend", "diffusion"),
        config.hypermog_run: ("HVFT", "hypermog"),
    }
    runs = load_requested_runs(csv_path, requested, config.sample_seed)
    thumbnails = {run_name: render_thumbnails(run, config.force_render) for run_name, run in runs.items()}

    comparisons = (
        (
            runs[config.diffusion_nft_run],
            f"greedy_hypervolume_4_subset_sample_{SAMPLE_SIZE}_seed_{config.sample_seed}_diffusion_nft_vs_hypermog.png",
        ),
        (
            runs[config.diffusion_blend_run],
            f"greedy_hypervolume_4_subset_sample_{SAMPLE_SIZE}_seed_{config.sample_seed}_diffusion_blend_vs_hypermog.pdf",
        ),
    )
    for diffusion_run, filename in comparisons:
        output_path = output_dir / filename
        plot_comparison(diffusion_run, runs[config.hypermog_run], thumbnails, output_path)
        print(f"Saved {diffusion_run.method} comparison to {output_path}")


if __name__ == "__main__":
    main()
