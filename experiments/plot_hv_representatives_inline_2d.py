import argparse
import os
import tempfile
from collections import deque
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flow-density-control-matplotlib"))

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

from plot_hv_representatives_2d import (
    COLORS,
    DEFAULT_DIFFUSION_BLEND_RUN,
    DEFAULT_DIFFUSION_NFT_RUN,
    DEFAULT_HYPERMOG_RUN,
    NUM_REPRESENTATIVES,
    SAMPLE_SIZE,
    RunRepresentatives,
    load_requested_runs,
    render_thumbnails,
    square_thumbnail,
)

# EDIT CARD POSITIONS HERE: centers in the initial axes coordinates, not QED/SA values.
# (0, 0) = bottom-left; (1, 1) = top-right. Increase x to move right, y to move up.
# Each method's cards are listed visually LEFT TO RIGHT in the current figure.
# Keys identify (method, molecule index); comments identify the printed scores.
# These presets reproduce the default runs with --sample-seed 3. Other runs/seeds
# use automatic placement. The frame expands around these positions as needed.
CARD_POSITIONS = {
    "DiffusionBlend": {
        # HVFT cards (above the curve), left to right:
        ("HVFT", 172): (0.180227, 0.777132),  # QED 0.392, SA 0.939
        ("HVFT", 517): (0.99, 0.35),  # QED 0.789, SA 0.708
        ("HVFT", 270): (0.865000, 0.705087),  # QED 0.775, SA 0.747
        ("HVFT", 46): (0.544024, 0.777132),  # QED 0.668, SA 0.818
        # DiffusionBlend cards (below the curve), left to right:
        ("DiffusionBlend", 120): (0.169337, 0.467802),  # QED 0.420, SA 0.928
        ("DiffusionBlend", 1069): (0.332946, 0.221786),  # QED 0.484, SA 0.804
        ("DiffusionBlend", 892): (0.593077, 0.269057),  # QED 0.575, SA 0.777
        ("DiffusionBlend", 1112): (0.848453, 0.131323),  # QED 0.729, SA 0.649
    },
    "DiffusionNFT": {
        # HVFT cards (above the curve), left to right:
        ("HVFT", 172): (0.180227, 0.777132),  # QED 0.392, SA 0.939
        ("HVFT", 517): (0.500000, 0.777132),  # QED 0.789, SA 0.708
        ("HVFT", 46): (0.765391, 0.592336),  # QED 0.668, SA 0.818
        ("HVFT", 270): (0.808167, 0.869509),  # QED 0.775, SA 0.747
        # DiffusionNFT cards (below the curve), left to right:
        ("DiffusionNFT", 110): (0.140821, 0.130491),  # QED 0.535, SA 0.805
        ("DiffusionNFT", 885): (0.225111, 0.453811),  # QED 0.420, SA 0.928
        ("DiffusionNFT", 846): (0.406290, 0.176680),  # QED 0.518, SA 0.841
        ("DiffusionNFT", 707): (0.674226, 0.117413),  # QED 0.641, SA 0.701
    },
}

# Physical sizes (points) keep cards consistent across thumbnail resolutions.
MOLECULE_WIDTH = 46
MOLECULE_HEIGHT = 32
CARD_PADDING = 2
# Base print size; the canvas grows if edited card positions exceed the frame.
FIGURE_SIZE = (3.25, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DiffusionNFT and DiffusionBlend representatives against HVFT with molecule images beside their points."
    )
    parser.add_argument("csv_path", type=Path, help="retro_compatibility_summary.csv path")
    parser.add_argument("--diffusion-nft-run", default=DEFAULT_DIFFUSION_NFT_RUN)
    parser.add_argument("--diffusion-blend-run", default=DEFAULT_DIFFUSION_BLEND_RUN)
    parser.add_argument("--hvft-run", default=DEFAULT_HYPERMOG_RUN)
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: beside the CSV)")
    parser.add_argument("--sample-seed", type=int, default=3, help="Random seed for reproducible 100-point sampling (default: 3, uses editable card positions)")
    parser.add_argument("--force-render", action="store_true", help="Regenerate individual molecule PNGs; the complete figure is always redrawn")
    return parser.parse_args()


def rectangles_overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    overlap_x = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap_y = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return overlap_x * overlap_y


@lru_cache(maxsize=None)
def transparent_thumbnail(image_path: Path) -> np.ndarray:
    rgb = np.asarray(square_thumbnail(image_path), dtype=np.uint8)
    background_candidate = np.all(rgb >= 246, axis=2)
    background = np.zeros(background_candidate.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    height, width = background.shape
    for row, column in (
        [(0, column) for column in range(width)]
        + [(height - 1, column) for column in range(width)]
        + [(row, 0) for row in range(1, height - 1)]
        + [(row, width - 1) for row in range(1, height - 1)]
    ):
        if background_candidate[row, column] and not background[row, column]:
            background[row, column] = True
            queue.append((row, column))
    while queue:
        row, column = queue.popleft()
        for next_row, next_column in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and background_candidate[next_row, next_column]
                and not background[next_row, next_column]
            ):
                background[next_row, next_column] = True
                queue.append((next_row, next_column))
    alpha = np.where(background, 0, 255).astype(np.uint8)
    return np.dstack((rgb, alpha))


def molecule_card(thumbnail_path: Path, qed: float, sa: float, color: str, scale: float = 1.0) -> VPacker:
    rgba = transparent_thumbnail(thumbnail_path)
    # Trim white margins without changing the molecule's aspect ratio.
    rows, columns = np.where(rgba[..., 3] > 0)
    if len(rows):
        rgba = rgba[max(0, rows.min() - 4) : rows.max() + 5, max(0, columns.min() - 4) : columns.max() + 5]
    height, width = rgba.shape[:2]
    image = OffsetImage(
        rgba,
        zoom=min(scale * MOLECULE_WIDTH / width, scale * MOLECULE_HEIGHT / height),
        resample=True,
        interpolation="hanning",
    )
    values = TextArea(
        f"QED  {qed:.3f}\nSA  {sa:.3f}",
        textprops={"fontsize": 7.5, "color": "black", "multialignment": "center"},
    )
    return VPacker(children=[image, values], align="center", pad=0, sep=2 * scale)


def choose_thumbnail_positions(
    objective_ax: plt.Axes,
    runs: tuple[RunRepresentatives, ...],
    card_sizes: dict[tuple[str, int], tuple[float, float]],
) -> dict[tuple[str, int], tuple[float, float]]:
    """Fit entire cards below diffusion curves and above the HVFT curve."""
    entries = []
    curves = {}
    sides = {}
    for run in runs:
        axes_points = objective_ax.transAxes.inverted().transform(objective_ax.transData.transform(run.values))
        curves[run.run] = axes_points[np.argsort(axes_points[:, 0])]
        sides[run.run] = run.side
        for index, point in enumerate(axes_points):
            entries.append(((run.run, index), point))
    entries.sort(key=lambda entry: float(entry[1][1]), reverse=True)
    all_points = np.asarray([point for _, point in entries])
    positions = {}
    occupied = {}
    margin = 0.018

    def rectangle(center, size):
        half_width, half_height = np.asarray(size) / 2
        return (center[0] - half_width, center[1] - half_height, center[0] + half_width, center[1] + half_height)

    # Revisit early cards once their neighbors are known.
    for _ in range(6):
        for key, point in entries:
            occupied.pop(key, None)
            half = np.asarray(card_sizes[key]) / 2
            low, high = half + margin, 1 - half - margin
            if np.any(low >= high):
                raise ValueError("Molecule cards exceed the axes; increase the figure size.")
            candidates = []
            for radius in (1.2, 1.8, 2.5):
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    offset = radius * (half + 0.025) * [np.cos(angle), np.sin(angle)]
                    candidates.append(np.clip(point + offset, low, high))
            # Grid candidates provide fallback space for clustered points.
            candidates.extend(
                np.array([x, y]) for x in np.linspace(low[0], high[0], 17) for y in np.linspace(low[1], high[1], 17)
            )
            if key in positions:
                candidates.insert(0, np.asarray(positions[key]))
            best_score = float("inf")
            best_center = None
            for center in candidates:
                box = rectangle(center, card_sizes[key])
                # Check the full card against the steps-post curve, including
                # vertical jumps. Extend endpoint heights beyond the curve ends.
                curve = curves[key[0]]
                edge_indices = np.clip(np.searchsorted(curve[:, 0], [box[0], box[2]], side="right") - 1, 0, len(curve) - 1)
                jumps = np.flatnonzero((curve[:, 0] >= box[0]) & (curve[:, 0] <= box[2]))
                heights = np.concatenate((curve[edge_indices, 1], curve[jumps, 1], curve[np.maximum(jumps - 1, 0), 1], [point[1]]))
                if sides[key[0]] == "diffusion" and box[3] > heights.min() - margin:
                    continue
                if sides[key[0]] == "hypermog" and box[1] < heights.max() + margin:
                    continue
                expanded = (box[0] - 0.009, box[1] - 0.012, box[2] + 0.009, box[3] + 0.012)
                overlap = sum(rectangles_overlap(expanded, other) for other in occupied.values())
                covered = np.count_nonzero(
                    (all_points[:, 0] >= box[0] - 0.014)
                    & (all_points[:, 0] <= box[2] + 0.014)
                    & (all_points[:, 1] >= box[1] - 0.018)
                    & (all_points[:, 1] <= box[3] + 0.018)
                )
                # Penalize connectors that pass through another card.
                leader = point + np.linspace(0.05, 0.95, 24)[:, None] * (center - point)
                crossings = sum(
                    np.any(
                        (leader[:, 0] >= other[0])
                        & (leader[:, 0] <= other[2])
                        & (leader[:, 1] >= other[1])
                        & (leader[:, 1] <= other[3])
                    )
                    for other in occupied.values()
                )
                distance = np.linalg.norm((center - point) * [1, 0.8])
                score = 10000 * overlap + 250 * covered + 0.35 * crossings + distance
                if score < best_score:
                    best_score, best_center, best_box = score, center, box
            if best_center is None:
                raise ValueError("No room for a molecule card on its required side of the curve; expand the axis limits.")
            positions[key] = tuple(best_center)
            occupied[key] = best_box
    return positions


def add_inline_thumbnails(
    objective_ax: plt.Axes,
    runs: tuple[RunRepresentatives, ...],
    thumbnails: dict[str, list[Path]],
    card_scales: dict[tuple[str, int], float] | None = None,
    fixed_positions: dict[tuple[str, int], tuple[float, float]] | None = None,
) -> None:
    fig = objective_ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cards, card_sizes = {}, {}
    padding_px = 2 * CARD_PADDING * fig.dpi / 72
    for run in runs:
        if len(thumbnails[run.run]) != len(run.values):
            raise ValueError(f"Thumbnail/value count mismatch for {run.run}")
        for index, ((qed, sa), path) in enumerate(zip(run.values, thumbnails[run.run])):
            key = (run.run, index)
            scale = card_scales.get((run.method, index), 1.0) if card_scales is not None else 1.0
            cards[key] = molecule_card(path, qed, sa, COLORS[run.side], scale)
            cards[key].set_figure(fig)
            bbox = cards[key].get_window_extent(renderer)
            card_sizes[key] = (
                (bbox.width + padding_px) / objective_ax.bbox.width,
                (bbox.height + padding_px) / objective_ax.bbox.height,
            )
    if fixed_positions is None:
        positions = choose_thumbnail_positions(objective_ax, runs, card_sizes)
    else:
        positions = {(run.run, index): fixed_positions[(run.method, molecule_index)] for run in runs for index, molecule_index in enumerate(run.representative_indices)}
    # Anchor cards to the initial data view, so expanding the limits brings both
    # cards and curves into the frame without changing the editable coordinates.
    positions = {key: tuple(objective_ax.transData.inverted().transform(objective_ax.transAxes.transform(center))) for key, center in positions.items()}
    annotations = []
    for run in runs:
        color = COLORS[run.side]
        for index, (qed, sa) in enumerate(run.values):
            key = (run.run, index)
            annotation = AnnotationBbox(
                cards[key],
                (float(qed), float(sa)),
                xybox=positions[key],
                xycoords="data",
                boxcoords="data",
                box_alignment=(0.5, 0.5),
                frameon=True,
                # AnnotationBbox padding is a fraction of its fontsize.
                fontsize=10,
                pad=CARD_PADDING / 10,
                bboxprops={
                    "boxstyle": f"round,pad={CARD_PADDING / 10},rounding_size=0.25",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.8,
                },
                annotation_clip=False,
                zorder=3,
            )
            objective_ax.add_artist(annotation)
            annotations.append((annotation, positions[key], (float(qed), float(sa)), color))
    # Draw every connector beneath every card, even when a leader crosses one.
    fig.canvas.draw()
    for annotation, center, point, color in annotations:
        objective_ax.add_artist(
            ConnectionPatch(
                xyA=center,
                xyB=point,
                coordsA=objective_ax.transData,
                coordsB=objective_ax.transData,
                axesA=objective_ax,
                axesB=objective_ax,
                patchA=None,
                shrinkA=0,
                shrinkB=2,
                arrowstyle="-",
                color=color,
                linewidth=0.95,
                zorder=2.5,
            )
        )

    # Expand the canvas and data limits together to preserve card spacing and
    # print sizes, instead of squeezing the existing layout into a smaller area.
    margin_px = 3 * fig.dpi / 72
    for _ in range(20):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = [annotation.patch.get_window_extent(renderer) for annotation, _, _, _ in annotations]
        frame = objective_ax.bbox
        left = min(frame.x0, min(box.x0 for box in bounds) - margin_px)
        right = max(frame.x1, max(box.x1 for box in bounds) + margin_px)
        bottom = min(frame.y0, min(box.y0 for box in bounds) - margin_px)
        top = max(frame.y1, max(box.y1 for box in bounds) + margin_px)
        if max(frame.x0 - left, right - frame.x1, frame.y0 - bottom, top - frame.y1) < 0.1:
            break
        lower, upper = objective_ax.transData.inverted().transform([(left, bottom), (right, top)])
        extra_width = (frame.x0 - left) + (right - frame.x1)
        extra_height = (frame.y0 - bottom) + (top - frame.y1)
        frame_x, frame_y = frame.x0, frame.y0
        frame_width, frame_height = frame.width + extra_width, frame.height + extra_height
        fig.set_size_inches(fig.get_size_inches() + np.array([extra_width, extra_height]) / fig.dpi)
        objective_ax.set_position([frame_x / fig.bbox.width, frame_y / fig.bbox.height, frame_width / fig.bbox.width, frame_height / fig.bbox.height])
        objective_ax.set_xlim(lower[0], upper[0])
        objective_ax.set_ylim(lower[1], upper[1])


def style_objective_axes(objective_ax: plt.Axes) -> None:
    axis_color = "black"
    objective_ax.set_xlabel("QED", fontsize=10, color=axis_color, labelpad=4)
    objective_ax.set_ylabel("Normalized SA score", fontsize=10, color=axis_color, labelpad=4)
    objective_ax.set_axisbelow(True)
    objective_ax.grid(False, which="both")
    objective_ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        objective_ax.spines[side].set_color("black")
        objective_ax.spines[side].set_linewidth(0.85)
    # Keep arrowheads at the axis ends when card fitting expands the limits.
    for endpoint, marker in (((1, 0), ">"), ((0, 1), "^")):
        objective_ax.plot(*endpoint, marker=marker, markersize=5, color=axis_color, transform=objective_ax.transAxes, clip_on=False, linestyle="none", markeredgewidth=0, zorder=6)
    for axis in (objective_ax.xaxis, objective_ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
        axis.set_minor_locator(AutoMinorLocator(2))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    objective_ax.tick_params(
        axis="both", which="major", direction="out", colors=axis_color, labelsize=9, length=3, width=0.8, pad=3
    )
    objective_ax.tick_params(axis="both", which="minor", direction="out", colors="black", length=2, width=0.6)


def plot_comparison(
    diffusion_run: RunRepresentatives,
    hvft_run: RunRepresentatives,
    thumbnails: dict[str, list[Path]],
    output_path: Path,
    sample_seed: int,
) -> None:
    with plt.rc_context(
        {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
         "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
         "xtick.color": "black", "ytick.color": "black", "pdf.fonttype": 42, "ps.fonttype": 42,
         "axes.unicode_minus": True}
    ):
        fig, objective_ax = plt.subplots(figsize=FIGURE_SIZE, dpi=150, facecolor="white")
        # Fixed margins preserve the requested physical size, including the legend.
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.91)
        runs = (diffusion_run, hvft_run)
        all_values = np.concatenate([run.values for run in runs], axis=0)
        if len(all_values) == 0 or not np.isfinite(all_values).all():
            plt.close(fig)
            raise ValueError("Representative values must be nonempty and finite.")
        spans = np.maximum(np.ptp(all_values, axis=0), 0.05)
        # Reserve vertical space for complete cards above and below the curves.
        padding = spans * [0.30, 0.65]
        objective_ax.set_xlim(all_values[:, 0].min() - padding[0], all_values[:, 0].max() + padding[0])
        objective_ax.set_ylim(all_values[:, 1].min() - padding[1], all_values[:, 1].max() + spans[1] * 0.95)
        style_objective_axes(objective_ax)

        for run in runs:
            color = COLORS[run.side]
            points = run.values[np.argsort(run.values[:, 0])]
            objective_ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=1.6,
                linestyle=(0, (5, 3)) if run.side == "diffusion" else "-",
                drawstyle="steps-post",
                zorder=2,
            )
            objective_ax.scatter(
                points[:, 0],
                points[:, 1],
                color=color,
                marker="D" if run.side == "hypermog" else "o",
                s=24,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
        handles = [
            Line2D(
                [0],
                [0],
                color=COLORS[run.side],
                linewidth=1.6,
                linestyle=(0, (10, 3)) if run.side == "diffusion" else "-",
                marker="D" if run.side == "hypermog" else "o",
                markersize=4.5,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=run.method,
            )
            for run in runs
        ]
        default_run = {"DiffusionBlend": DEFAULT_DIFFUSION_BLEND_RUN, "DiffusionNFT": DEFAULT_DIFFUSION_NFT_RUN}.get(diffusion_run.method)
        use_preset = sample_seed == 3 and diffusion_run.run == default_run and hvft_run.run == DEFAULT_HYPERMOG_RUN
        fixed_positions = CARD_POSITIONS[diffusion_run.method] if use_preset else None
        add_inline_thumbnails(objective_ax, runs, thumbnails, fixed_positions=fixed_positions)
        # Trim the unused top of the axis without changing point or card spacing.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        cards = [artist for artist in objective_ax.artists if isinstance(artist, AnnotationBbox)]
        content_top = max(card.patch.get_window_extent(renderer).y1 for card in cards)
        upper = objective_ax.transData.inverted().transform((objective_ax.bbox.x0, content_top + 4 * fig.dpi / 72))[1]
        lower, previous_upper = objective_ax.get_ylim()
        upper = min(upper, previous_upper)
        frame = objective_ax.get_position()
        objective_ax.set_ylim(lower, upper)
        objective_ax.set_position([frame.x0, frame.y0, frame.width, frame.height * (upper - lower) / (previous_upper - lower)])
        # Align the legend entries with QED, with DiffusionBlend at the far left.
        fig.canvas.draw()
        label_box = objective_ax.xaxis.label.get_window_extent(fig.canvas.get_renderer())
        legend_y = (label_box.y0 + label_box.y1) / (2 * fig.bbox.height)
        for handle, location, x in zip(handles, ("center left", "center right"), (0.0, frame.x1)):
            fig.legend(handles=[handle], loc=location, bbox_to_anchor=(x, legend_y), frameon=False, fontsize=9, handlelength=2.0, borderaxespad=0, labelcolor="black")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Always rebuild and overwrite the figure, even when molecule PNGs are reused.
        fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(output_path.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def main() -> None:
    config = parse_args()
    csv_path = config.csv_path.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve() if config.output_dir is not None else csv_path.parent
    requested = {
        config.diffusion_nft_run: ("DiffusionNFT", "diffusion"),
        config.diffusion_blend_run: ("DiffusionBlend", "diffusion"),
        config.hvft_run: ("HVFT", "hypermog"),
    }
    runs = load_requested_runs(csv_path, requested, config.sample_seed)
    thumbnails = {run_name: render_thumbnails(run, config.force_render) for run_name, run in runs.items()}
    transparent_thumbnail.cache_clear()
    comparisons = (
        (
            runs[config.diffusion_blend_run],
            f"greedy_hypervolume_{NUM_REPRESENTATIVES}_subset_sample_{SAMPLE_SIZE}_seed_{config.sample_seed}_diffusion_blend_vs_hvft_inline.pdf",
        ),
    )
    for diffusion_run, filename in comparisons:
        output_path = output_dir / filename
        plot_comparison(diffusion_run, runs[config.hvft_run], thumbnails, output_path, config.sample_seed)
        print(f"Saved {diffusion_run.method} inline comparison to {output_path}")


if __name__ == "__main__":
    main()
