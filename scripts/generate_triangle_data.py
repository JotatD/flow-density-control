"""Generate uniform triangle decisions and their maximization objectives."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusiongym import DDTensor

from genexp.mo import TriangleReward


def sample_triangle(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    points = rng.random((num_samples, 2))
    # Folding the upper half of the square preserves uniform area density.
    outside = points.sum(axis=1) > 1.0
    points[outside] = 1.0 - points[outside]
    return points.astype(np.float32)

def generate_data(output_dir: Path, num_samples: int = 42000, seed: int = 42):
    rng = np.random.default_rng(seed)
    training = sample_triangle(num_samples, rng)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward = TriangleReward()
    sample = DDTensor(torch.from_numpy(training))
    with torch.no_grad():
        values, _ = reward(sample, sample)
    np.save(output_dir / "dec.npy", training)
    np.save(output_dir / "obj.npy", values.cpu().numpy())
    (output_dir / "metadata.json").write_text(json.dumps({
        "seed": seed, "training_samples": num_samples,
        "training_region": "x >= 0, y >= 0, x + y <= 1", "coordinates": "raw (x, y)",
        "domain": [[0.0, 1.0], [0.0, 1.0]],
        "objectives": ["x + y", "-(x + y - 1)^2"], "maximize": [True, True],
    }, indent=2) + "\n")
    print(f"Saved {num_samples} triangle decisions and objectives to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("assets/triangle"))
    parser.add_argument("--num_samples", type=int, default=42000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_data(**vars(args))
