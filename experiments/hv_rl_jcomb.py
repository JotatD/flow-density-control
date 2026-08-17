"""Tune PepTune with DMPO using final permeability as the Optuna objective."""

from __future__ import annotations

import argparse
import os
import pickle as pkl
import random
import traceback
from pathlib import Path
from typing import Any, final

import numpy as np
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from peptidesgym import DiscreteEnvironment, JCombModel
from peptidesgym.rewards import CombinatorialReward

from genexp.mo.base import MOReward
from genexp.mo.utils import (
    HVComputer,
    calculate_reward_bin_percentages,
    plot_clipped_values,
    plot_reward_bin_progression,
    plot_score_density,
)
from genexp.trainers.hv_discrete_rl import HVDiscreteRL
from genexp.wandb_log import WandbLogger

import datetime

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune PepTune with DMPO and final permeability.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--name", type=str, default="jcomb2", help="Optuna study and W&B project name.")
    parser.add_argument("--optuna_seed", type=int, default=42, help="Random seed for the Optuna sampler.")
    args = parser.parse_args()
    return args


def reward_distribution_score(percentages):
    p = np.asarray(percentages, dtype=float)
    p = p / p.sum()

    num_good = len(p) - 1
    ideal = np.concatenate(
        [
            np.full(num_good, 1.0 / num_good),
            [0.0],
        ]
    )

    total_variation = 0.5 * np.abs(p - ideal).sum()
    return 1.0 - total_variation
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
    reward: CombinatorialMO
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

class CombinatorialMO(CombinatorialReward, MOReward):
    def __init__(self, *args, **kwargs):
        MOReward.__init__(self, torch.tensor([-1.0, -1.0]), 2)
        CombinatorialReward.__init__(self, *args, **kwargs)
    
    def __call__(self, sample: Any, latent: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        reward_value, info = CombinatorialReward.__call__(self, sample, latent, **kwargs)
        return reward_value, info

def main(config: DictConfig) -> float:
    print(config, flush=True)
    root = Path(__file__).resolve().parents[1]
    torch.hub.set_dir(str(root / ".torch-cache" / "hub"))
    seed_everything(int(config.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_path = Path(f"output/{config.project_name}/{config.run_name}")
    os.makedirs(output_path, exist_ok=True)
    config.num_md_iterations = 1000//config.trainer.max_epochs

    log = WandbLogger(
        project_name=config.project_name,
        config=OmegaConf.to_container(config, resolve=True),  # type: ignore
        use_wandb=bool(config.wandb),
        run_name=config.run_name,
    )
    global_step = log.set_step_metric(0, "global_step")
    md_iteration = log.set_step_metric(0, "md_iteration")
    loss = log.watch("loss", "global_step")
    eval_img = log.set_image("eval_image", "global_step")
    reward_progress_img = log.set_image("reward_bin_progression", "global_step")
    reward_med = log.watch("reward_median", "global_step")
    full_hv = log.watch("full_hypervolume", "global_step")
    n_hv = log.watch("n_hypervolume", "global_step")
    dtst_img = log.set_image("dataset_image", "global_step")
    tv = log.watch("1_m_total_variation", "global_step")
    

    #try
    base_model = JCombModel(config=config, device=device).eval()
    fine_model = JCombModel(config=config, device=device)
    fine_model.load_state_dict(base_model.state_dict())

    hv_computer = HVComputer(ref_point=torch.tensor([-1.0, -1.0]), num_rew=2)
    
    reward = CombinatorialMO(device=device)
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
    
    # inner_ckpt_queue = CheckpointQueue(max_size=3, name_prefix="jcomb", save_dir=output_path)
    # dataset_ckpt_queue = CheckpointQueue(max_size=3, name_prefix="dataset", save_dir=output_path)
    # outer_ckpt_queue = CheckpointQueue(max_size=2, name_prefix="outer", save_dir=output_path)
    
    last_model_path = Path(
        "/home/juan.guevara/test/tr-counter/TR2-D2/tr2d2-jcomb/outputs/bounded_uniform_binary32_seed42/pretraining/pretrained_best.ckpt"
    )
    fine_model.diffusion_model.load_state_dict(torch.load(last_model_path, map_location=device))
    base_model.diffusion_model.load_state_dict(torch.load(last_model_path, map_location=device))
    dataset = None
    max_epochs = int(config.trainer.max_epochs)
    resample_every = int(config.resample_every)
    reward_percentage_history = []
    # try:
    for _ in range(config.num_md_iterations):
        md_iteration += 1
        for epoch in range(max_epochs):
            global_step += 1
            print(f"Epoch {epoch + 1}/{max_epochs} of MD iteration {md_iteration.val}", flush=True)
            if dataset is None or epoch % resample_every == 0:
                print(f"Sampling new dataset for MD iteration {md_iteration.val}, epoch {epoch + 1}", flush=True)
                dataset = trainer.generate_dataset()

            rewards = evaluate(trainer, reward=reward, num_samples=2048, batch_size=2048)

            should_plot = global_step.val % 100 == 0
            if should_plot:
                eval_img.val = plot_score_density(rewards.numpy(), plot_path=output_path, filename=f"eval_scores_md_{md_iteration.val}_epoch_{epoch + 1}.png")
            pctgs = calculate_reward_bin_percentages(rewards, reward.reward_vectors)
            tv.val = reward_distribution_score(pctgs)
            reward_percentage_history.append(pctgs)
            if should_plot:
                reward_progress_img.val = plot_reward_bin_progression(
                    reward_percentage_history,
                    reward.reward_vectors,
                    output_path,
                )
            reward_med.val = np.median(rewards.numpy())
            full_hv.val = hv_computer(rewards.reshape(1, -1, 2)).item()
            n_hv.val = hv_computer(rewards.reshape(-1, config.n, 2)).mean().item()
            print(f"Median reward: {reward_med.val}, Full hypervolume: {full_hv.val}, N hypervolume: {n_hv.val}", flush=True)
                
                #dataset_ckpt_queue.add(dataset)
                
            loss.val = trainer.finetune(dataset)
            #inner_ckpt_queue.add(trainer.fine_model)
        trainer.update_base_model()
        #outer_ckpt_queue.add(trainer.base_model)
        torch.save(trainer.fine_model.state_dict(), output_path / "final_model.pt")

    final_rewards = evaluate(
        trainer,
        reward=reward,
        num_samples=2048,
        batch_size=2048,
    )
    pctgs = calculate_reward_bin_percentages(final_rewards, reward.reward_vectors)
    print(f"Final reward percentages: {pctgs}", flush=True)
    reward_percentage_history.append(pctgs)
    tv.val = reward_distribution_score(pctgs)
    reward_progress_img.val = plot_reward_bin_progression(
        reward_percentage_history,
        reward.reward_vectors,
        output_path,
    )
    eval_img.val = plot_score_density(
        final_rewards.numpy(), plot_path=output_path, filename=f"eval_scores_md_{md_iteration.val}_epoch_{epoch + 1}.png"
    )
    log.finish()
    return tv.val


def optuna_entry(trial: optuna.Trial) -> float:
    args = parse_args()
    x = 1
    config = {
        "n": trial.suggest_categorical("n", [4, 8, 16, 32]),
        "temperature": 1e-5,
        "num_lambda": 400,
        "num_p_nm1": 256 // x,
        "sample_p_nm1_batch_size": 256,
        "vol_samples": 256 // x,
        "noise": {
            "type": "loglinear",
        },
        # Legacy aliases used by diffusion/MCTS code.
        "eps": 1e-3,
        "noise_eps": 1e-3,
        "lmbda": 1,
        "dmpo": {
            "lr": trial.suggest_float(
                "dmpo_lr",
                1e-4,
                1e-3,
                log=True,
            ),
            "batch_size": 2048,
            "alpha": trial.suggest_categorical(
                "dmpo_alpha",
                [3e-5, 0.1, 6.0],
            ),
            "importance_coefficient": 1.0,
            "clip_grad_norm": trial.suggest_categorical(
                "clip_grad_norm",
                [2.0, -1],
            ),
            "num_replicates": 16,
            "mask_eps": 1e-3,
            "centering": trial.suggest_categorical(
                "centering",
                [False, True],
            ),
            "num_samples": 100 // x,
        },
        "sampling": {
            "predictor": "ddpm_cache",
            "seq_length": 32,
            "steps": 32,
            "noise_removal": True,
        },
        # Inner optimization loop.
        "trainer": {
            "max_epochs": trial.suggest_categorical(
                "max_epochs",
                [20, 32],
            ),
        },
        "resample_every": trial.suggest_categorical(
            "resample_every",
            [8, 10, 20, 32],
        ),
        # Outer optimization remains fixed at 1000 steps.
        # "outer_gradient_steps": 1000,
        "time_conditioning": False,
        "T": 0,
        "seed": 42,
        "num_eval_samples": 2048,
        "training": {
            "antithetic_sampling": True,
            "sampling_eps": 1e-3,
        },
        "eval": {
            "gen_ppl_eval_model_name_or_path": "gpt2-large",
            "perplexity_batch_size": 8,
        },
        "optim": {
            "weight_decay": 0.075,
            "lr": 3e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        "lr_scheduler": {
            "num_warmup_steps": 2500,
        },
        "model": {
            "length": 32,
        },
        "roformer": {
            "hidden_size": 128,
            "n_layers": 4,
            "n_heads": 4,
            "max_position_embeddings": 32,
        },
        "wandb": args.wandb,
        "project_name": args.name,
        "run_name": (f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d_%H-%M-%S}_trial-{trial.number}"),
    }
    resolved = OmegaConf.create(config)
    return main(resolved)


if __name__ == "__main__":
    cli_args = parse_args()
    n_jobs = 1  # Number of parallel jobs for Optuna optimization
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=20,
        n_ei_candidates=48,
        multivariate=True,
        group=True,
        constant_liar=n_jobs > 1,
    )

    study = optuna.create_study(
        study_name=cli_args.name,
        sampler=sampler,
        direction="maximize",
        storage="sqlite:///optuna_store.db",
        load_if_exists=True,
    )
    study.optimize(
        optuna_entry,
        n_trials=200,
        n_jobs=n_jobs,
        gc_after_trial=True,
    )
