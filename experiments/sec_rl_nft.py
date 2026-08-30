import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from diffusiongym import Sample
from diffusiongym.environments import EndpointEnvironment
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.flowmol import GEOMBaseModel
from diffusiongym.rewards import DummyReward, Reward
from torch.utils.hipify.hipify_python import str2bool
from tqdm.auto import tqdm
from utils import seed_everything

from genexp.mo.mo_mol import RDkitReward
from genexp.resume import (
    load_latest_training_checkpoint,
    mark_run_complete,
    resolve_run,
    restore_rng_state,
    save_training_checkpoint,
)
from genexp.trainers.hv_rl import DiffusionNFTrainer
from genexp.trainers.utils import StepTimer
from genexp.wandb_log import WandbLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    #loggin
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--force_new_start", action="store_true")
    parser.add_argument("--project_name", type=str, default="one_fv")
    parser.add_argument("--run_name", type=str, default="hv_nft")
    parser.add_argument("--seed", type=int, default=5)

    #algorithm and problem
    parser.add_argument("--scalarization", type=str, choices=["sum", "improvement"], default="improvement")
    parser.add_argument("--reward", type=str, choices=["qed", "sa"], default="qed")
    
    #hyper param
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--adaptive_loss_scaling", type=str2bool, default="y")
    parser.add_argument("--adaptive_scale_eps", type=float, default=1e-5)
    parser.add_argument("--num_inner_epochs", type=int, default=1) # currently unused
    parser.add_argument("--num_integration_steps", type=int, default=100)

    #every n step
    parser.add_argument("--update_pretrained_every_n_steps", type=int, default=1000)
    parser.add_argument("--sample_nm1_every_n_steps", type=int, default=1000)
    parser.add_argument("--resample_every_n_steps", type=int, default=1)
    
    parser.add_argument("--save_every_n_steps", type=int, default=10)
    
    #size
    parser.add_argument("--backward_batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--advantage_group_size", type=int, default=32)
    parser.add_argument("--num_p_nm1", type=int, default=60)
    parser.add_argument("--vol_samples", type=int, default=64)
    parser.add_argument("--num_diversity_samples", type=int, default=128)
    parser.add_argument("--timestep_fraction", type=float, default=1.0)

    parser.add_argument("--fulfill_num_samples", type=str2bool, default="n")
    parser.add_argument("--only_valids", type=str2bool, default="n")
    parser.add_argument("--fulfill_max_attempts", type=int, default=10_000)
    
    #eval every
    parser.add_argument("--evaluate_diversity_every_n_steps", type=int, default=20)
    parser.add_argument("--evaluate_every_n_steps", type=int, default=1)
    
    #modifiers
    parser.add_argument("--validate_2d", type=str, default="none", choices=["none", "full"])
    parser.add_argument("--validate_3d", type=str, default="none", choices=["none", "fast", "full"])
    parser.add_argument("--exploration_decay_type", type=int, choices=[0, 1, 2], default=1)
    
    parser.add_argument("--fixed_A", type=int, default=10) # if 0, the atoms are selcted at random
    parser.add_argument("--invalid_val", type=float, default=-1.0)

    #logging
    parser.add_argument("--num-time-groups", type=int, default=2)    
    
    return parser.parse_args()
    
def sample_x(num_samples: int, trainer: DiffusionNFTrainer, discretization_steps: int = 128) -> list[Sample[DDGraph]]:
    left = num_samples
    samples: list[Sample] = []

    original_policy = trainer.env._policy
    trainer.env.policy = trainer.fine_model
    try:
        with torch.no_grad():
            while left > 0:
                batch = min(left, trainer.config.batch_size)
                sample = trainer.env.sample(batch, discretization_steps=discretization_steps, pbar=True)
                samples.extend([s for s in sample])
                left -= batch
    finally:
        trainer.env._policy = original_policy
    
    return samples



def evaluate(samples: list[Sample], reward: Reward) -> tuple[torch.Tensor, float]:
    samples_cat = Sample.concat(samples)
    rew, info = reward(samples_cat.sample, samples_cat.latent)
    valids = info["valids"].sum().item()
    rewards = [r for i, r in enumerate(rew) if info["valids"][i]]

    reward_values = torch.stack(rewards, dim=0)

    return reward_values, valids / len(samples)


def summarize_rewards(rewards: torch.Tensor) -> tuple[float, float, float]:
    rew_mean = rewards.mean().item()
    rew_med = rewards.median().item()
    rew_std = rewards.std().item()
    print(rewards)

    return rew_mean, rew_med, rew_std


def main(config: Namespace) -> None:
    assert config.sample_nm1_every_n_steps % config.resample_every_n_steps == 0
    assert config.update_pretrained_every_n_steps % config.resample_every_n_steps == 0
    assert config.update_pretrained_every_n_steps % config.sample_nm1_every_n_steps == 0
    assert (not config.fulfill_num_samples and not config.only_valids) or (config.fulfill_num_samples and config.only_valids)
    timer = StepTimer(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with timer.section("setup"):
        results_root = Path("output") / config.project_name
        run_resolution = resolve_run(config, results_root, config.run_name)
        if run_resolution.completed:
            print(f"Matching run is already complete: {run_resolution.run_dir}")
            return

        print(f"run_dir={run_resolution.run_dir}")
        log = WandbLogger(
            project_name=config.project_name,
            config=vars(config),
            use_wandb=config.wandb,
            run_name=run_resolution.run_dir.name,
            id=run_resolution.wandb_run_id,
            resume="allow",
            dir=str(run_resolution.run_dir),
        )

        seed_everything(int(config.seed))

        print("problem=dxtb_10A")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rew_cnf = {"valid_3d": config.validate_3d, "valid_2d": config.validate_2d, "invalid_val": config.invalid_val}
        if config.reward == "sa":
            reward = RDkitReward(rewards=["sa"], **rew_cnf)
        elif config.reward == "qed":
            reward = RDkitReward(rewards=["qed"], **rew_cnf)
        else:
            raise ValueError(f"Unknown reward: {config.reward}")
        
        model = GEOMBaseModel(device=device)
        env = EndpointEnvironment(model, DummyReward(), discretization_steps=int(config.num_integration_steps))
        if config.fixed_A > 0:
            unconstrained_sample = env.sample
            env.sample = lambda *args, **kwargs: unconstrained_sample(*args, n_atoms=config.fixed_A, **kwargs)  # ty: ignore[invalid-assignment]


        trainer = DiffusionNFTrainer(config, env, base_model=model, device=device)

        start_epoch = -1
        dataset = None
        advantages = None
    
    with timer.section("load_checkpoint"):
        resume_checkpoint = load_latest_training_checkpoint(run_resolution.run_dir, map_location=device)
        if resume_checkpoint is not None:
                trainer.load_training_state_dict(resume_checkpoint["trainer_state"])
                start_epoch = resume_checkpoint["next_epoch"]
                loop_state = resume_checkpoint.get("loop_state", {})
                dataset = loop_state.get("dataset")
                advantages = loop_state.get("advantages")
                restore_rng_state(resume_checkpoint)

    vol_samples = config.vol_samples

    epoch = log.set_step_metric(start_epoch, "epoch")

    rew_mean = log.watch(f"{config.reward}_mean", "epoch")
    rew_med = log.watch(f"{config.reward}_median", "epoch")
    rew_std = log.watch(f"{config.reward}_std", "epoch")

    valid_frac = log.watch("valid_fraction", "epoch")
    # valid_3d = log.watch("diversity/validity_3d", "epoch")
    # diversity_usrcat = log.watch("diversity/diversity_usrcat", "epoch")
    # vendi_usrcat = log.watch("diversity/vendi_usrcat", "epoch")
    # auc_usrcat = log.watch("diversity/auc_coverage_usrcat", "epoch")
    # valid_2d = log.watch("diversity/validity_2d", "epoch")
    # diversity_tanimoto = log.watch("diversity/diversity_tanimoto", "epoch")
    # vendi_tanimoto = log.watch("diversity/vendi_tanimoto", "epoch")
    # auc_tanimoto = log.watch("diversity/auc_coverage_tanimoto", "epoch")
    fulfillment = log.watch("dataset/fulfillment", "epoch")
    first_var = log.watch("dataset/median_first_var", "epoch")

    # if epoch.val == -1:
    #     samples_eval = sample_x(vol_samples, trainer, discretization_steps=config.num_integration_steps)
    #     first_val_eval.val, n_hv.val, full_hv.val, rewards, valid_frac.val = evaluate(trainer, samples_eval, hv_computer, n=config.n)
    #     (qed.val, sa.val), (qed_td.val, sa_td.val), (qed_t3.val, sa_t3.val) = summarize_rewards(rewards)
    
    
    for _ in tqdm(range(start_epoch, config.epochs)):
        epoch += 1
        
        # if epoch.val % config.evaluate_diversity_every_n_steps == 0:
        #     with timer.section("evaluate_diversity"):
        #         if config.validate_2d != "none" or config.validate_3d != "none":

        #             samples_diversity = sample_x(config.num_diversity_samples, trainer, discretization_steps=config.num_integration_steps)
        #             sample = Sample.concat(samples_diversity).sample
        #             mols: list[Chem.Mol] = graph_to_mols(sample)

        #             valid_2d_count, diversity_tanimoto.val, vendi_tanimoto.val, auc_tanimoto.val = diversity_metrics_2d(mols)
        #             valid_2d.val = valid_2d_count / len(mols)

        #             full_bust = (config.validate_3d == "full")
        #             valid_3d_count, diversity_usrcat.val, vendi_usrcat.val, auc_usrcat.val = diversity_metrics_3d(mols, full_bust=full_bust)
        #             valid_3d.val = valid_3d_count / len(mols)


        if epoch.val % config.resample_every_n_steps == 0:
            with timer.section("generate_dataset"):
                trainer.update_exploration_model()
                dataset, advantages, fv = trainer.generate_dataset(reward)
                
                fulfillment.val = len(dataset) / config.num_samples
    
                if dataset:
                    first_var.val = torch.median(fv).item()     
                       
        if epoch.val % config.save_every_n_steps == 0:
            save_training_checkpoint(
                run_resolution.run_dir,
                next_epoch=epoch.val,
                trainer_state=trainer.training_state_dict(),
                loop_state={
                    "dataset": dataset,
                    "advantages": advantages,
                }
            )
            torch.save(trainer.fine_model.state_dict(), run_resolution.run_dir / f"model_epoch_{epoch.val}.pt")

        if not dataset or advantages is None:
            print("No valid dataset or advantages, skipping epoch.")
            continue
        
        print(f"Epoch {epoch.val} starting with dataset size: {len(dataset)} and advantages size: {len(advantages)}")
        with timer.section("finetune"):
            timed_stats = trainer.finetune(dataset, advantages)
            
        
        fulltime_stats = {f"full/{k}": np.mean(v) for k, v in timed_stats.items()}
        log.log_dict(fulltime_stats, 'epoch')
        
        trainer.update_off_policy_model()
                    
        rows = timer.summary()
        
        if epoch.val % config.evaluate_every_n_steps == 0:
            with timer.section("evaluate_hypervolume"):
                samples_eval = sample_x(vol_samples, trainer, discretization_steps=config.num_integration_steps)
                rewards, valid_frac.val = evaluate(samples_eval, reward)
                rew_mean.val, rew_med.val, rew_std.val = summarize_rewards(rewards)

        print("\n=== Timing summary (by total time) ===")
        for name, cnt, total, mean, p50, p95 in rows:
            print(f"{name:30s}  n={cnt:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  "
                f"p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")

        print(f"Epoch {epoch.val} completed")

    log.finish()
    mark_run_complete(run_resolution.run_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
