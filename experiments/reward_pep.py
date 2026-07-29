#!/usr/bin/env python
"""Tune PepTune with DMPO using final permeability as the Optuna objective."""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from diffusiongym.rewards import Reward
from diffusiongym.types import DDTensor
from omegaconf import DictConfig, OmegaConf

from genexp.trainers.dmpo import DMPOTrainer
from genexp.wandb_log import WandbLogger
from peptidesgym import DiscreteEnvironment, PeptuneModel, String
from peptidesgym.rewards import PermeabilityReward
from genexp.mo.utils import plot_clipped_values
import os


class DummyPeptideReward(Reward[DDTensor]):
    """Zero reward used while the environment collects DMPO trajectories."""

    def __call__(self, sample: String, latent: DDTensor, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.zeros(len(sample), device=latent.device), {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune PepTune with DMPO and final permeability.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--name", type=str, default="reward_pep_study", help="Optuna study and W&B project name.")
    parser.add_argument("--optuna_seed", type=int, default=42, help="Random seed for the Optuna sampler.")
    args = parser.parse_args()
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_permeability(
    trainer: DMPOTrainer,
    permeability: PermeabilityReward,
    num_samples: int,
    batch_size: int,
) -> torch.Tensor:
    rewards = []
    trainer.fine_model.eval()
    for start in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - start)
        env_sample = trainer.env.sample(current_batch_size, pbar=False)
        reward_output = permeability(env_sample.sample, env_sample.latent)
        reward_values = reward_output[0] if isinstance(reward_output, tuple) else reward_output
        rewards.append(torch.as_tensor(reward_values).detach().float().cpu())

    result = torch.cat(rewards)
    return result


def main(config: DictConfig) -> float:
    root = Path(__file__).resolve().parents[1]
    torch.hub.set_dir(str(root / ".torch-cache" / "hub"))
    seed_everything(int(config.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log = WandbLogger(
        project_name=config.project_name,
        config=OmegaConf.to_container(config, resolve=True), # type: ignore
        use_wandb=bool(config.wandb),
        run_name=config.run_name,
    )
    epoch_step = log.set_step_metric(0, "epoch")
    global_step = log.set_step_metric(0, "global_step")
    loss = log.watch("loss", "global_step")
    grad_norm = log.watch("grad_norm", "global_step")
    mean_reward = log.watch("mean_reward", "global_step")
    mean_lrmp = log.watch("mean_log_reference_minus_policy", "global_step")
    sample_eff = log.watch("effective_sample_size", "global_step")
    data_img = log.set_image("data_image", "epoch")

    try:
        print(f"Loading PepTune reference and policy models on {device}")
        base_model = PeptuneModel(config=config, device=device).eval()
        fine_model = PeptuneModel(config=config, device=device)

        environment = DiscreteEnvironment(base_model=fine_model, reward=DummyPeptideReward(), discretization_steps=int(config.sampling.steps))
        permeability = PermeabilityReward(device=device)
        trainer = DMPOTrainer(
            config=config,
            env=environment,
            fine_model=fine_model,
            base_model=base_model,
            reward_fn=permeability,
            device=device,
            verbose=True,
        )

        dataset = None
        max_epochs = int(config.trainer.max_epochs)
        resample_every = int(config.resample_every)
        for epoch in range(max_epochs):
            epoch_step.val = epoch + 1
            if dataset is None or epoch % resample_every == 0:
                print(f"Epoch {epoch + 1}/{max_epochs}: generating fresh rollouts")
                dataset = trainer.generate_dataset()
                dtst_rews = torch.cat([d.full_env_sample.rewards for d in dataset], dim=0).detach().cpu().numpy()
                data_img.val = plot_clipped_values(high=0.0, low=-10.0, values=dtst_rews)

            loss.val = trainer.finetune(dataset)
            global_step += 1
            grad_norm.val = trainer.last_metrics["grad_norm"]
            mean_reward.val = trainer.last_metrics["mean_reward"]
            mean_lrmp.val = trainer.last_metrics["mean_log_reference_minus_policy"]
            sample_eff.val = trainer.last_metrics["effective_sample_size"]

        rewards = evaluate_permeability(trainer, permeability, num_samples=int(config.num_eval_samples), batch_size=int(config.batch_size))

        output_path = Path(f"output/{config.project_name}/{config.run_name}")
        os.makedirs(output_path, exist_ok=True)
        torch.save(trainer.fine_model.state_dict(), output_path / "final_model.pt")
        return rewards.mean().item()
    finally:
        log.finish()


def optuna_entry(trial: optuna.Trial) -> float:
    args = parse_args()
    config = {
        "noise": {
            "type": "loglinear",
            "sigma_min": 1e-4,
            "sigma_max": 20,
            "state_dependent": True,
        },
        "mode": "train",
        "diffusion": "absorbing_state",
        "vocab": "old_smiles",
        "backbone": "finetune_roformer",
        "parameterization": "subs",
        "time_conditioning": False,
        "T": 0,
        "subs_masking": False,
        "seed": 42,
        "checkpoint": "../peptidesgym/checkpoints/peptune-pretrained.ckpt",
        "lr": trial.suggest_categorical("lr", [5e-6, 1e-5, 2e-5]),
        "batch_size": 8,
        "num_replicates": 16,
        "resample_every": trial.suggest_categorical("resample_every", [1, 4, 8]),
        "alpha": trial.suggest_categorical("alpha", [0.5, 1.0, 2.0]),
        "importance_coefficient": 1.0,
        "clip_grad_norm": 2.0,
        "mask_eps": 1e-3,
        "centering": False,
        "num_eval_samples": 64,
        "sampling": {
            "predictor": "ddpm_cache",
            "num_samples": 64,
            "sampling_eps": 1e-3,
            "steps": 16,
            "seq_length": 32,
            "noise_removal": True,
        },
        "training": {
            "sampling_eps": 1e-3,
        },
        "eval": {
            "checkpoint_path": None,
            "gen_ppl_eval_model_name_or_path": "gpt2-large",
            "perplexity_batch_size": 8,
        },
        "optim": {
            "lr": 1e-5,
        },
        "roformer": {
            "hidden_size": 768,
            "n_layers": 8,
            "n_heads": 8,
            "max_position_embeddings": 1035,
        },
        "trainer": {
            "max_epochs": 100,
        },
        "wandb": args.wandb,
        "project_name": trial.study.study_name,
        "run_name": f"trial_{trial.number}",
    }
    resolved = OmegaConf.create(config)
    return main(resolved)


if __name__ == "__main__":
    cli_args = parse_args()
    study = optuna.create_study(
        study_name=cli_args.name,
        sampler=optuna.samplers.BruteForceSampler(seed=cli_args.optuna_seed),
        direction="maximize",
        storage=f"sqlite:///optuna_store.db",
        load_if_exists=True,
    )
    study.optimize(optuna_entry, n_trials=9)
