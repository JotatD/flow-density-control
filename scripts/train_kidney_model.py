"""Pretrain an epsilon diffusion model on raw red-kidney coordinates."""

import argparse
from pathlib import Path

import torch
import numpy as np
from diffusiongym.environments import EpsilonEnvironment
from diffusiongym.schedulers import DiffusionScheduler
from matplotlib.figure import Figure
from pytorch_lightning import seed_everything
from train_zdt_model import load_training_data, train_diffusion

from genexp.mo import KidneyReward
from genexp.base_models.mlp import TensorMLPModel


def save_samples_and_plot(model, output_dir: Path, num_samples: int, sample_steps: int, batch_size: int):
    # Lightning teardown can move weights to CPU while model.device stays on CUDA.
    model.to(model.device)
    model.eval()
    env = EpsilonEnvironment(model, KidneyReward(), discretization_steps=sample_steps)
    batches = []
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            result = env.sample(min(batch_size, num_samples - start), pbar=False)
            batches.append(result.sample.data.cpu())
    points = torch.cat(batches).numpy()
    np.save(output_dir / "sampled_dec.npy", points)

    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.35, color="tab:red", label="Diffusion samples")
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    ax.plot(np.cos(theta) - 0.45 * np.cos(2.0 * theta), 1.3 * np.sin(theta), color="black", linewidth=1.5, label="Red kidney boundary")
    ax.set(xlabel="x", ylabel="y", title="Samples from the trained kidney diffusion model", aspect="equal")
    ax.legend()
    fig.savefig(output_dir / "sampled_points.png", dpi=200)
    print(f"Saved {num_samples} diffusion samples and sampled_points.png to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data_path", type=Path, default=Path("assets/kidney/dec.npy"))
    parser.add_argument("--output_model_path", type=Path, default=Path("assets/kidney/pretrained_diffusion.pth"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of post-training diffusion samples")
    parser.add_argument("--sample_steps", type=int, default=1000, help="Diffusion integration steps for sampling")
    parser.add_argument("--sample_only", action="store_true", help="Sample and plot the saved checkpoint without retraining")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_epochs <= 0:
        parser.error("batch_size and max_epochs must be positive")
    if args.num_samples <= 0 or args.sample_steps <= 0:
        parser.error("num_samples and sample_steps must be positive")
    seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.sample_only:
        # Match train_diffusion's default noise schedule and architecture.
        betas = torch.linspace(0.1 / 1000, 12.0 / 1000, 1000, device=device)
        scheduler = DiffusionScheduler((1.0 - betas).cumprod(dim=0))
        model = TensorMLPModel(scheduler, output_type="epsilon", input_dim=2, device=device).to(device)
        model.model.load_state_dict(torch.load(args.output_model_path, map_location=device, weights_only=True))
        print(f"Loaded pretrained diffusion model from {args.output_model_path}")
    else:
        data = load_training_data(args.train_data_path)
        if data.shape[1] != 2 or data.shape[0] == 0:
            raise ValueError("Expected nonempty red-kidney data with shape (N, 2)")
        x, y = data.unbind(dim=1)
        q1 = (x + 0.45 * (1.0 - 2.0 * (y / 1.3).square())).square() + (y / 1.3).square()
        if (q1 > 1.0 + 1e-6).any():
            raise ValueError("Training data contains points outside the red kidney")
        args.output_model_path.parent.mkdir(parents=True, exist_ok=True)
        model = train_diffusion(data, args.output_model_path, device, args.batch_size, args.max_epochs)
        print(f"Saved pretrained diffusion model to {args.output_model_path}")
    save_samples_and_plot(model, args.output_model_path.parent, args.num_samples, args.sample_steps, args.batch_size)


if __name__ == "__main__":
    main()
