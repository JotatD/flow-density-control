"""Tune PepTune with DMPO using final permeability as the Optuna objective."""

from __future__ import annotations

import argparse
import os
import pickle as pkl
import random
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from peptidesgym import DiscreteEnvironment, PeptuneModel

from genexp.mo.mo_pep import PeptideMOReward
from genexp.mo.utils import HVComputer, plot_clipped_values
from genexp.trainers.hv_discrete_rl import HVDiscreteRL
from genexp.wandb_log import WandbLogger


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


class CheckpointQueue:
    def __init__(self, max_size: int, name_prefix: str, save_dir: Path):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        self.max_size = max_size
        self.name_prefix = name_prefix
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, index: int, extension: str) -> Path:
        suffix = "last" if index == 0 else f"last_minus_{index}"
        return self.save_dir / f"{self.name_prefix}_{suffix}.{extension}"

    def add(self, obj: torch.nn.Module | Any) -> Path:
        extension = "pt" if isinstance(obj, torch.nn.Module) else "pkl"

        # Remove the oldest checkpoint if the queue is full.
        for ext in ("pt", "pkl"):
            oldest = self._checkpoint_path(self.max_size - 1, ext)
            oldest.unlink(missing_ok=True)

        # Shift existing checkpoints backward:
        # last -> last_minus_1, last_minus_1 -> last_minus_2, etc.
        for index in range(self.max_size - 2, -1, -1):
            for ext in ("pt", "pkl"):
                source = self._checkpoint_path(index, ext)

                if source.exists():
                    destination = self._checkpoint_path(index + 1, ext)
                    source.replace(destination)

        checkpoint_path = self._checkpoint_path(0, extension)

        if isinstance(obj, torch.nn.Module):
            torch.save(obj.state_dict(), checkpoint_path)
        else:
            with checkpoint_path.open("wb") as file:
                pkl.dump(obj, file)

        return checkpoint_path

@torch.no_grad()
def evaluate(
    trainer: HVDiscreteRL,
    num_samples: int,
    batch_size: int,
    reward: PeptideMOReward,
) -> torch.Tensor:
    rewards = []
    trainer.fine_model.eval()
    env_model = trainer.env.base_model
    env_reward = trainer.env.reward
    trainer.env.base_model = trainer.fine_model
    trainer.env.reward = reward
    for start in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - start)
        env_sample = trainer.env.sample(current_batch_size, pbar=False)
        rewards.append(env_sample.rewards.detach().float().cpu())
    trainer.env.base_model = env_model
    trainer.env.reward = env_reward 
    result = torch.cat(rewards)
    return result


def compute_eval_hypervolumes(
    rewards: torch.Tensor,
    group_size: int,
    reward: PeptideMOReward,
    hv_computer: HVComputer,
) -> tuple[float, float]:
    """Compute full and grouped hypervolumes from the available rewards."""
    if rewards.ndim != 2 or rewards.shape[1] != reward.num_rew:
        raise ValueError(
            f"Expected rewards with shape (num_samples, {reward.num_rew}), got {tuple(rewards.shape)}"
        )
    if group_size < 1:
        raise ValueError("group_size must be at least 1")

    num_grouped_samples = (rewards.shape[0] // group_size) * group_size
    if num_grouped_samples == 0:
        raise ValueError(
            f"Need at least {group_size} reward samples to compute grouped hypervolume; got {rewards.shape[0]}"
        )

    full_hv = hv_computer(rewards.unsqueeze(0)).item()
    grouped_rewards = rewards[:num_grouped_samples].reshape(-1, group_size, reward.num_rew)
    grouped_hv = hv_computer(grouped_rewards).mean().item()
    return full_hv, grouped_hv


def main(config: DictConfig) -> float:
    root = Path(__file__).resolve().parents[1]
    torch.hub.set_dir(str(root / ".torch-cache" / "hub"))
    seed_everything(int(config.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_path = Path(f"output/{config.project_name}/{config.run_name}")
    os.makedirs(output_path, exist_ok=True)

    log = WandbLogger(
        project_name=config.project_name,
        config=OmegaConf.to_container(config, resolve=True),  # type: ignore
        use_wandb=bool(config.wandb),
        run_name=config.run_name,
    )
    global_step = log.set_step_metric(0, "global_step")
    md_iteration = log.set_step_metric(0, "md_iteration")
    loss = log.watch("loss", "global_step")
    full_hv = log.watch("full_hypervolume", "global_step")
    n_hv = log.watch("n_hypervolume", "global_step")
    dtst_img = log.set_image("dataset_image", "global_step")
    # ambient = np.load('assets/pep_200/data/obj.npy')


    #try
    print(f"Loading PepTune reference and policy models on {device}")
    base_model = PeptuneModel(config=config, device=device).eval()
    fine_model = PeptuneModel(config=config, device=device)

    reward = PeptideMOReward(
        reward_names=["permeability", "hemolysis"],
        zero_invalid=True,
        device=device,
    )
    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)
    environment = DiscreteEnvironment(
        base_model=fine_model,
        reward=reward,
        discretization_steps=int(config.sampling.steps),
    )
    trainer = HVDiscreteRL(
        config=config,
        env=environment,
        base_model=base_model,
        fine_model=fine_model,
        device=device,
        verbose=True,
    )
    
    reward_trackers = {name: log.watch(f"{name}", "global_step") for name in reward.reward_names}

    inner_ckpt_queue = CheckpointQueue(max_size=3, name_prefix="peptune", save_dir=output_path)
    dataset_ckpt_queue = CheckpointQueue(max_size=3, name_prefix="dataset", save_dir=output_path)
    outer_ckpt_queue = CheckpointQueue(max_size=2, name_prefix="outer", save_dir=output_path)
    
    # last_model_path = Path('/shared/home/juan.guevara/Code/flow-density-control/output/first_hv_pep/trial_0/final_model.pt')
    # fine_model.load_state_dict(torch.load(last_model_path, map_location=device))
    dataset = None
    max_epochs = int(config.trainer.max_epochs)
    resample_every = int(config.resample_every)
    try:
        for _ in range(config.num_md_iterations):
            md_iteration += 1
            for epoch in range(max_epochs):
                global_step += 1
                print(f"Epoch {epoch + 1}/{max_epochs} of MD iteration {md_iteration.val}")
                if dataset is None or epoch % resample_every == 0:
                    print(f"Sampling new dataset for MD iteration {md_iteration.val}, epoch {epoch + 1}")
                    dataset = trainer.generate_dataset()
                    dtst_img.val = plot_clipped_values(low=-1, high=80, values=dataset.rewards.detach().cpu().numpy())

                    rewards = evaluate(trainer, reward=reward, num_samples=config.num_eval_samples, batch_size=config.dmpo.batch_size)
                    for i, name in enumerate(reward.reward_names):
                        reward_trackers[name].val = rewards[:, i].mean().item()
                    full_hv.val, n_hv.val = compute_eval_hypervolumes(rewards=rewards, group_size=int(config.n), reward=reward, hv_computer=hv_computer)
                    
                    dataset_ckpt_queue.add(dataset)
                    
                loss.val = trainer.finetune(dataset)
                inner_ckpt_queue.add(trainer.fine_model)
            trainer.update_base_model()
            outer_ckpt_queue.add(trainer.base_model)
            torch.save(trainer.fine_model.state_dict(), output_path / "final_model.pt")
        return rewards.mean().item()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"An error occurred: {e}")
    finally:
        with open(output_path / "lasty_last_dataset.pkl", "wb") as f:
            pkl.dump(dataset, f)
        torch.save(trainer.fine_model.state_dict(), output_path / "finally_final_model.pt")
        log.finish()


def optuna_entry(trial: optuna.Trial) -> float:
    args = parse_args()
    x = 1
    config = {
        "n": 4,
        "num_md_iterations": 15,
        "temperature": 1e-5,
        "num_lambda": 400,
        "num_p_nm1": 256 // x,
        "sample_p_nm1_batch_size": 40,
        "vol_samples": 256 // x,
        "noise": {
            "type": "loglinear",
            "sigma_min": 1e-4,
            "sigma_max": 20,
            "state_dependent": True,
        },
        "lmbda": 1,
        "dmpo": {
            "lr": trial.suggest_categorical("lr", [1e-4]),
            "batch_size": 40,
            "alpha": trial.suggest_categorical("alpha", [0.1]),
            "importance_coefficient": 1.0,
            "clip_grad_norm": -1.0,
            "num_replicates": 16,  # at replication time, the num_samples size is multiplied by this number
            "mask_eps": 1e-3,
            "centering": False,
            "num_samples": 100 // x,  # num samples in the dataset
        },
        "sampling": {
            "predictor": "ddpm_cache",
            "seq_length": 200 // x,  # important parameter
            "sampling_eps": 1e-3,
            "steps": 128 // x,  # important
            "noise_removal": True,
        },
        "trainer": {
            "max_epochs": 32,
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
        "checkpoint": "/home/juan.guevara/test/peptune/peptidesgym/checkpoints/peptune-pretrained.ckpt",
        "resample_every": trial.suggest_categorical("resample_every", [8]),
        "num_eval_samples": 64,
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
        storage="sqlite:///optuna_store.db",
        load_if_exists=True,
    )
    study.optimize(optuna_entry, n_trials=9)
