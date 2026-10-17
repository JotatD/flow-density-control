"""Generate uniform red-kidney decisions and their maximization objectives."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusiongym import DDTensor

from genexp.mo import KidneyReward


def sample_red_kidney(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    radius = np.sqrt(rng.random(num_samples))
    theta = rng.uniform(0.0, 2.0 * np.pi, num_samples)
    a, b = radius * np.cos(theta), radius * np.sin(theta)
    # Constant Jacobian 1.3 preserves uniform area density.
    return np.column_stack((a - 0.45 * (1.0 - 2.0 * b**2), 1.3 * b)).astype(np.float32)


def generate_data(output_dir: Path, num_samples: int = 42000, seed: int = 42):
    rng = np.random.default_rng(seed)
    training = sample_red_kidney(num_samples, rng)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward = KidneyReward()
    sample = DDTensor(torch.from_numpy(training))
    with torch.no_grad():
        values, _ = reward(sample, sample)
    np.save(output_dir / "dec.npy", training)
    np.save(output_dir / "obj.npy", values.cpu().numpy())
    (output_dir / "metadata.json").write_text(json.dumps({
        "seed": seed, "training_samples": num_samples,
        "training_region": "Q_red <= 1", "coordinates": "raw (x, y)",
        "domain": [[-1.8, 4.15], [-1.65, 1.65]],
        "objectives": ["f1", "f2"], "maximize": [True, True],
    }, indent=2) + "\n")
    print(f"Saved {num_samples} red-kidney decisions and objectives to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("assets/kidney"))
    parser.add_argument("--num_samples", type=int, default=42000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_data(**vars(args))
