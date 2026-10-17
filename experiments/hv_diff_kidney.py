"""Hypervolume-guided diffusion optimization from red-kidney pretraining."""

import argparse
import json
import time
from math import isfinite
from pathlib import Path

import numpy as np
import torch
from diffusiongym.environments import EpsilonEnvironment
from diffusiongym.schedulers import DiffusionScheduler
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from utils import seed_everything

from genexp.base_models.mlp import TensorMLPModel
from genexp.mo import KidneyReward
from genexp.mo.utils import HVComputer
from genexp.trainers.hv_diff import HVDiff
from genexp.wandb_log import WandbLogger

DECISION_BOUNDS = np.array([[-1.8, 4.15], [-1.65, 1.65]])
OBJECTIVE_BOUNDS = np.array([[0.0, 5.0], [0.0, 5.0]])


def expanded_bounds(bounds, *point_sets):
    """Keep baseline limits exactly, extending only edges exceeded by points."""
    limits = np.array(bounds, dtype=float, copy=True)
    for points in point_sets:
        for axis in range(2):
            values = points[:, axis]
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            padding = 0.03 * (bounds[axis, 1] - bounds[axis, 0])
            if values.min() < bounds[axis, 0]:
                limits[axis, 0] = min(limits[axis, 0], values.min() - padding)
            if values.max() > bounds[axis, 1]:
                limits[axis, 1] = max(limits[axis, 1], values.max() + padding)
    return limits


def plot_points(training, sampled, objective_space=False):
    fig = Figure(figsize=(8, 6), layout="constrained")
    ax = fig.subplots()
    ax.scatter(training[:, 0], training[:, 1], s=5, alpha=0.12, color="tab:red", label="Red-kidney training data")
    ax.scatter(sampled[:, 0], sampled[:, 1], s=18, alpha=0.8, color="tab:blue", label="Current diffusion samples")
    if objective_space:
        t = np.linspace(-1.0, 1.0, 500)
        ax.plot(1.0 + (1.0 + t)**2, 1.0 + (1.0 - t)**2, color="black", label="Pareto front")
        bounds = OBJECTIVE_BOUNDS
        ax.set(xlabel="Objective 1 (maximize)", ylabel="Objective 2 (maximize)", title="Objective space")
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, 500)
        ax.plot(np.cos(theta) - 0.45 * np.cos(2.0 * theta), 1.3 * np.sin(theta), color="black", label="Red kidney boundary")
        bounds = DECISION_BOUNDS
        ax.set(xlabel="x", ylabel="y", title="Decision space")
    limits = expanded_bounds(bounds, training, sampled)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    return fig


def build_diffusion_model(config, device):
    n_steps = int(config.diffusion_steps)
    betas = torch.linspace(config.beta_0 / n_steps, config.beta_1 / n_steps, n_steps, device=device)
    scheduler = DiffusionScheduler((1.0 - betas).cumprod(dim=0))
    model = TensorMLPModel(scheduler, output_type="epsilon", input_dim=2, device=device).to(device)
    model.model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
    return model.eval()


def evaluate_hypervolume(trainer, num_samples, hv_computer, sampling_steps):
    decisions, objectives = [], []
    old_steps = trainer.env.discretization_steps
    trainer.env.discretization_steps = sampling_steps
    try:
        with torch.no_grad():
            collected = 0
            while collected < num_samples:
                sample = trainer.sample_trajectories()
                count = min(len(sample.sample), num_samples - collected)
                decisions.append(sample.sample.data[:count].detach().cpu())
                objectives.append(sample.rewards[:count].to(trainer.device))
                collected += count
    finally:
        trainer.env.discretization_steps = old_steps
    values = torch.cat(objectives)
    if not torch.isfinite(values).all() or not torch.isfinite(torch.cat(decisions)).all():
        raise ValueError("Evaluation produced non-finite decisions or objectives")
    n_hv = hv_computer(values.reshape(-1, trainer.n, 2)).mean().item()
    full_hv = hv_computer(values.unsqueeze(0)).item()
    return n_hv, full_hv, torch.cat(decisions).numpy(), values.detach().cpu().numpy()


def main(config: DictConfig):
    if config.n < 2 or config.vol_samples <= 0 or config.vol_samples % config.n:
        raise ValueError("n must be at least 2 and vol_samples a positive multiple of n")
    seed_everything(int(config.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(config.data_dir)
    training_dec = np.load(data_dir / "dec.npy")
    training_obj = np.load(data_dir / "obj.npy")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    reward = KidneyReward()
    model = build_diffusion_model(config, device)
    env = EpsilonEnvironment(model, reward, discretization_steps=int(config.adjoint_matching.sampling.num_integration_steps))
    trainer = HVDiff(config, env, device=device)
    hv_computer = HVComputer(ref_point=reward.ref_point.to(device), num_rew=2)
    log = WandbLogger(config=OmegaConf.to_container(config, resolve=True), use_wandb=config.wandb, project_name=config.project_name, run_name=config.run_name)
    global_step = log.set_step_metric(0, "global_step")
    md_step = log.set_step_metric(0, "md_step")
    n_hv_metric = log.watch("n_hypervolume", "global_step")
    full_hv_metric = log.watch("full_hypervolume", "global_step")
    loss_metric = log.watch("loss", "global_step")
    decision_image = log.set_image("decision_points", "global_step")
    objective_image = log.set_image("objective_points", "global_step")
    history = []

    def evaluate_and_save(loss=None):
        n_hv, full_hv, decisions, objectives = evaluate_hypervolume(trainer, int(config.vol_samples), hv_computer, int(config.eval_steps))
        n_hv_metric.val, full_hv_metric.val = n_hv, full_hv
        step = global_step.val
        history.append({"global_step": step, "md_step": md_step.val, "loss": loss, "n_hypervolume": n_hv, "full_hypervolume": full_hv})
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        np.savez(output_dir / f"samples_{step:04d}.npz", dec=decisions, obj=objectives)
        for name, training, sampled, objective_space, metric in (
            ("decision", training_dec, decisions, False, decision_image),
            ("objective", training_obj, objectives, True, objective_image),
        ):
            fig = plot_points(training, sampled, objective_space)
            path = output_dir / f"{name}_{step:04d}.png"
            fig.savefig(path, dpi=160)
            if config.wandb:
                metric.val = str(path)
            fig.clear()
        print(f"step={step} n_hypervolume={n_hv:.6f} full_hypervolume={full_hv:.6f}", flush=True)

    try:
        evaluate_and_save()
        for _ in range(config.num_md_iterations):
            md_step += 1
            for _ in range(config.adjoint_matching.num_iterations):
                global_step += 1
                dataset = trainer.generate_dataset()
                loss = float(trainer.finetune(dataset, steps=None))
                if not isfinite(loss):
                    raise ValueError("Non-finite adjoint matching loss")
                loss_metric.val = loss
                evaluate_and_save(loss)
            trainer.update_base_model()
            torch.save(trainer.fine_model.model.state_dict(), output_dir / "finetuned_diffusion.pth")
    finally:
        log.finish()


def default_config():
    run_name = f"kidney_{time.strftime('%Y%m%d_%H%M%S')}"
    return OmegaConf.create({
        "seed": 5, "n": 100, "num_md_iterations": 10,
        "alpha_div": 1e-4, "lmbda": 1e3, "temperature": 1e-5,
        "num_lambda": 4000, "num_p_nm1": 768, "sample_p_nm1_batch_size": 64,
        "vol_samples": 500, "eval_steps": 200,
        "diffusion_steps": 1000, "beta_0": 0.1, "beta_1": 12.0,
        "data_dir": "assets/kidney", "model_path": "assets/kidney/pretrained_diffusion.pth",
        "output_dir": f"output/{run_name}", "project_name": "kidney",
        "run_name": run_name, "wandb": False,
        "adjoint_matching": {
            "num_iterations": 30, "batch_size": 64, "clip_grad_norm": 2.0,
            "clip_loss": -1, "lr": 1e-5,
            "sampling": {"num_samples": 64, "num_integration_steps": 100},
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML overrides for experiment settings")
    parser.add_argument("--wandb", action="store_true", help="Enable online logging")
    args = parser.parse_args()
    config = default_config()
    if args.config is not None:
        config = OmegaConf.merge(config, OmegaConf.load(args.config))
    if args.wandb:
        config.wandb = True
    main(config)
