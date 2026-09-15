import html
import math
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import py3Dmol
from PIL import Image
from rdkit import Chem


EXAMPLE_PICKLE = Path(
    "/home/juan.guevara/Code/flow-density-control/output/geom_10_di_sec/"
    "hv_nft_20260904_231008_912634/checkpoints/mols_epoch_300.pkl"
)
CPK_COLORS = {
    "H": "#E8E8E8",
    "C": "#4A5568",
    "N": "#3B5BDB",
    "O": "#E03131",
    "F": "#2F9E44",
    "P": "#F08C00",
    "S": "#FAB005",
    "Cl": "#37B24D",
    "Br": "#A61E4D",
    "I": "#7048E8",
}
BACKGROUND_COLOR = "#FFFFFF"
VIEWER_WIDTH = 960
VIEWER_HEIGHT = 720
GRID_CELL_WIDTH = 520
GRID_CELL_HEIGHT = 420
GRID_HEADER_HEIGHT = 64


def _find_browser() -> str:
    for browser_name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        browser = shutil.which(browser_name)
        if browser is not None:
            return browser
    raise RuntimeError("A Chrome or Chromium executable is required to save the 3Dmol view as a PNG")


def _validate_rendered_image(image_path: Path) -> None:
    if not image_path.is_file() or image_path.stat().st_size == 0:
        raise RuntimeError("Headless Chrome did not create the rendered PNG")
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        if all(low == high for low, high in rgb_image.getextrema()):
            raise RuntimeError("The rendered PNG is blank; 3Dmol.js may not have loaded")


def _capture_viewer(html_path: Path, output_path: Path, width: int, height: int) -> None:
    browser = _find_browser()
    with tempfile.TemporaryDirectory(prefix="molecule-render-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        screenshot_path = temporary_path / "molecule.png"
        profile_path = temporary_path / "chrome-profile"

        command = [
            browser,
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--window-size={width},{height}",
            "--virtual-time-budget=5000",
            f"--user-data-dir={profile_path}",
            f"--screenshot={screenshot_path}",
            html_path.as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unknown browser error"
            raise RuntimeError(f"Could not capture the 3Dmol rendering: {message}")

        _validate_rendered_image(screenshot_path)
        shutil.copyfile(screenshot_path, output_path)


def _orient_for_rendering(molecule: Chem.Mol, positions: np.ndarray) -> Chem.Mol:
    centered_positions = positions - positions.mean(axis=0)
    covariance = centered_positions.T @ centered_positions
    _, principal_axes = np.linalg.eigh(covariance)
    oriented_positions = centered_positions @ principal_axes[:, ::-1]
    for axis in (0, 1):
        if np.sum(oriented_positions[:, axis] ** 3) < 0.0:
            oriented_positions[:, axis] *= -1.0

    oriented_molecule = Chem.Mol(molecule)
    conformer = oriented_molecule.GetConformer(0)
    for atom_index, position in enumerate(oriented_positions):
        conformer.SetAtomPosition(atom_index, tuple(float(value) for value in position))
    return oriented_molecule


def _prepare_molecule(molecule: Chem.Mol) -> Chem.Mol:
    if molecule.GetNumAtoms() == 0:
        raise ValueError("Cannot render a molecule without atoms")
    if molecule.GetNumConformers() == 0:
        raise ValueError("Cannot render a molecule without a stored conformer")

    positions = np.asarray(molecule.GetConformer(0).GetPositions(), dtype=float)
    if positions.shape != (molecule.GetNumAtoms(), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("The molecule conformer does not contain finite three-dimensional coordinates")
    return _orient_for_rendering(molecule, positions)


def _style_molecule_viewer(viewer: py3Dmol.view, viewer_position: tuple[int, int] | None = None) -> None:
    viewer_kwargs = {} if viewer_position is None else {"viewer": viewer_position}
    viewer.setStyle(
        {},
        {
            "stick": {"radius": 0.105, "colorscheme": {"prop": "elem", "map": CPK_COLORS}},
            "sphere": {"scale": 0.24, "colorscheme": {"prop": "elem", "map": CPK_COLORS}},
        },
        **viewer_kwargs,
    )
    viewer.setViewStyle(
        {
            "ambientOcclusion": {"strength": 1.4, "radius": 4.5},
            "outline": {"color": "#4B5563", "width": 0.002, "maxpixels": 1},
        },
        **viewer_kwargs,
    )
    viewer.setProjection("perspective", **viewer_kwargs)
    viewer.setBackgroundColor(BACKGROUND_COLOR, **viewer_kwargs)
    viewer.zoomTo(viewer=viewer_position) if viewer_position is not None else viewer.zoomTo()
    viewer.rotate(14, "x", **viewer_kwargs)
    viewer.rotate(-18, "y", **viewer_kwargs)
    viewer.rotate(6, "z", **viewer_kwargs)
    viewer.zoom(1.22, **viewer_kwargs)


def render_molecule_3d(molecule: Chem.Mol, output_path: str | Path) -> Path:
    oriented_molecule = _prepare_molecule(molecule)
    molecule_block = Chem.MolToMolBlock(oriented_molecule, confId=0)
    viewer = py3Dmol.view(width=VIEWER_WIDTH, height=VIEWER_HEIGHT)
    viewer.addModel(molecule_block, "mol")
    _style_molecule_viewer(viewer)
    viewer.render()

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    html_destination = destination.with_suffix(".html")
    viewer.write_html(str(html_destination))
    _capture_viewer(html_destination, destination, VIEWER_WIDTH, VIEWER_HEIGHT)
    return destination


def render_molecule_grid(
    molecules: Sequence[Chem.Mol], output_path: str | Path, labels: Sequence[str], title: str
) -> Path:
    if not molecules:
        raise ValueError("Cannot render an empty molecule grid")
    if len(molecules) != len(labels):
        raise ValueError(f"Expected one label per molecule, got {len(labels)} labels for {len(molecules)} molecules")

    columns = min(3, math.ceil(math.sqrt(len(molecules))))
    rows = math.ceil(len(molecules) / columns)
    viewer_width = columns * GRID_CELL_WIDTH
    viewer_height = rows * GRID_CELL_HEIGHT
    page_height = viewer_height + GRID_HEADER_HEIGHT
    viewer = py3Dmol.view(width=viewer_width, height=viewer_height, viewergrid=(rows, columns), linked=False)

    for index, molecule in enumerate(molecules):
        viewer_position = divmod(index, columns)
        molecule_block = Chem.MolToMolBlock(_prepare_molecule(molecule), confId=0)
        viewer.addModel(molecule_block, "mol", viewer=viewer_position)
        _style_molecule_viewer(viewer, viewer_position)
    viewer.render()

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    html_destination = destination.with_suffix(".html")
    viewer.write_html(str(html_destination))

    label_elements = []
    for index, label in enumerate(labels):
        row, column = divmod(index, columns)
        escaped_label = html.escape(label).replace("\n", "<br>")
        left = column * GRID_CELL_WIDTH + 16
        top = GRID_HEADER_HEIGHT + row * GRID_CELL_HEIGHT + 14
        label_elements.append(
            f'<div class="molecule-label" style="left:{left}px;top:{top}px;">{escaped_label}</div>'
        )

    overlay = f"""
<style>
html, body {{ width: {viewer_width}px; height: {page_height}px; margin: 0; overflow: hidden; background: #FFFFFF; }}
body {{ position: relative; font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
.grid-title {{ height: {GRID_HEADER_HEIGHT}px; box-sizing: border-box; display: flex; align-items: center;
  justify-content: center; color: #1F2937; font-size: 18px; font-weight: 650; letter-spacing: 0.01em;
  border-bottom: 1px solid #E5E7EB; background: #FFFFFF; }}
.molecule-label {{ position: absolute; z-index: 10; box-sizing: border-box; min-width: 250px; padding: 7px 11px;
  border: 1px solid rgba(209, 213, 219, 0.9); border-radius: 8px; color: #374151; background: rgba(255,255,255,0.92);
  box-shadow: 0 2px 8px rgba(17,24,39,0.08); font-size: 13px; line-height: 1.4; pointer-events: none; }}
</style>
<div class="grid-title">{html.escape(title)}</div>
{''.join(label_elements)}
"""
    html_text = html_destination.read_text()
    html_text = html_text.replace("<body style=\"margin: 0; padding: 0; display: block;\">", "<body>", 1)
    html_text = html_text.replace("<body>", f"<body>{overlay}", 1)
    html_destination.write_text(html_text)
    _capture_viewer(html_destination, destination, viewer_width, page_height)
    return destination


def main() -> None:
    try:
        with EXAMPLE_PICKLE.open("rb") as molecule_file:
            molecules = pickle.load(molecule_file)
    except Exception as exc:
        raise ValueError(f"Could not load molecules from {EXAMPLE_PICKLE}: {exc}") from exc

    if not isinstance(molecules, list):
        raise ValueError(f"Expected {EXAMPLE_PICKLE} to contain a list, got {type(molecules).__name__}")
    if not molecules:
        raise ValueError(f"No molecules found in {EXAMPLE_PICKLE}")

    output_path = EXAMPLE_PICKLE.with_name(f"{EXAMPLE_PICKLE.stem}_first_molecule_3d.png")
    rendered_path = render_molecule_3d(molecules[0], output_path)
    print(f"Saved 3D molecule rendering to {rendered_path}")
    print(f"Saved interactive 3D viewer to {rendered_path.with_suffix('.html')}")


if __name__ == "__main__":
    main()
