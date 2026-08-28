import argparse
from argparse import Namespace
from math import ceil
from pathlib import Path

import numpy as np
import torch
from diffusiongym import Sample
from diffusiongym.environments import EndpointEnvironment
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.flowmol import GEOMBaseModel
from diffusiongym.molecules.rewards.utils import graph_to_mols
from diffusiongym.rewards import DummyReward
from rdkit import Chem
from torch.utils.hipify.hipify_python import str2bool
from tqdm.auto import tqdm
from utils import seed_everything

from genexp.mo.base import MOReward
from genexp.mo.mo_mol import TopologyMetrics
from genexp.mo.moses import diversity_metrics_2d, diversity_metrics_3d
from genexp.mo.utils import HVComputer
from genexp.resume import (
    load_latest_training_checkpoint,
    mark_run_complete,
    resolve_run,
    restore_rng_state,
    save_training_checkpoint,
)
from genexp.trainers.hv_rl import HVRL
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
    parser.add_argument("--reward", type=str, choices=["topology"], default="topology")
    
    #hyper param
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_inner_epochs", type=int, default=1) # currently unused
    parser.add_argument("--num_integration_steps", type=int, default=100)

    #every n step
    parser.add_argument("--update_pretrained_every_n_steps", type=int, default=1000)
    parser.add_argument("--sample_nm1_every_n_steps", type=int, default=1000)
    parser.add_argument("--resample_every_n_steps", type=int, default=10)
    
    parser.add_argument("--save_every_n_steps", type=int, default=10)
    
    #size
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--advantage_group_size", type=int, default=64)
    parser.add_argument("--num_p_nm1", type=int, default=60)
    parser.add_argument("--vol_samples", type=int, default=64)
    parser.add_argument("--num_diversity_samples", type=int, default=128)
    parser.add_argument("--timestep_fraction", type=float, default=0.50)

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

    #logging
    parser.add_argument("--num-time-groups", type=int, default=2)    
    
    return parser.parse_args()

def sample_x(num_samples: int, trainer: HVRL, discretization_steps: int = 128) -> list[Sample[DDGraph]]:
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



def evaluate(trainer: HVRL, samples: list[Sample], hv_computer: HVComputer, n: int) -> tuple[torch.Tensor, float, float, torch.Tensor, float]:
    samples_cat = Sample.concat(samples)
    first_variation, info = trainer.hv_first_variation(samples_cat.sample, samples_cat.latent)
    valids = info["valids"].sum().item()
    rew = info["obj"].reshape(-1, trainer.reward.num_rew)
    rewards = [r for i, r in enumerate(rew) if info["valids"][i]]
    
    num_rew = trainer.reward.num_rew
    ref_point = trainer.reward.ref_point
        
    if rewards:
        reward_values = torch.stack(rewards, dim=0)
        full_objectives = reward_values.reshape(1, -1, num_rew)
        full_hypervolume = hv_computer(full_objectives).detach().cpu().item()
    else:
        reward_values = torch.zeros((1, num_rew), device=ref_point.device, dtype=torch.float32)
        full_hypervolume = 0.0
        
    as_many = reward_values.shape[0] - (reward_values.shape[0] % n)
    if as_many > 0:
        n_objectives = reward_values[:as_many].reshape(-1, n, num_rew)
        n_hypervolume = hv_computer(n_objectives).mean().detach().cpu().item()
    else: 
        n_hypervolume = 0.0
        
    return first_variation, n_hypervolume, full_hypervolume, reward_values, valids / len(samples)


def summarize_rewards(rewards: torch.Tensor) -> tuple[list[float], list[float], list[float]]:
    top_decile_count = ceil(rewards.shape[0] * 0.1)
    means = rewards.mean(dim=0)
    top_decile_means = rewards.topk(top_decile_count, dim=0).values.mean(dim=0)
    top_3_means = rewards.topk(min(3, rewards.shape[0]), dim=0).values.mean(dim=0)
    return tuple(values.detach().cpu().numpy().tolist() for values in (means, top_decile_means, top_3_means))


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
        reward = TopologyMetrics(valid_3d=config.validate_3d, valid_2d=config.validate_2d)
        model = GEOMBaseModel(device=device)
        env = EndpointEnvironment(model, DummyReward(), discretization_steps=int(config.num_integration_steps))

        hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)
        trainer = HVRL(config, env, reward, hv_computer=hv_computer, device=device)

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

    n_hv = log.watch("n_hypervolume", "epoch")
    full_hv = log.watch("full_hypervolume", "epoch")
    qed = log.watch("qed", "epoch")
    qed_td = log.watch("top_decile/qed_td", "epoch")
    qed_t3 = log.watch("top_3/qed_t3", "epoch")
    sa = log.watch("sa", "epoch")
    sa_td = log.watch("top_decile/sa_td", "epoch")
    sa_t3 = log.watch("top_3/sa_t3", "epoch")
    valid_frac = log.watch("valid_fraction", "epoch")
    
    # valid_2d = log.watch("diversity/validity_2d", "epoch")
    # valid_3d = log.watch("diversity/validity_3d", "epoch")
    # diversity_usrcat = log.watch("diversity/diversity_usrcat", "epoch")
    # vendi_usrcat = log.watch("diversity/vendi_usrcat", "epoch")
    # auc_usrcat = log.watch("diversity/auc_coverage_usrcat", "epoch")
    # diversity_tanimoto = log.watch("diversity/diversity_tanimoto", "epoch")
    # vendi_tanimoto = log.watch("diversity/vendi_tanimoto", "epoch")
    # auc_tanimoto = log.watch("diversity/auc_coverage_tanimoto", "epoch")
    
    first_var = log.watch(f"dataset/{config.scalarization}", "epoch")
    first_val_median = log.watch(f"{config.scalarization}_median", "epoch")
    first_val_mean = log.watch(f"{config.scalarization}_mean", "epoch")
    fulfillment = log.watch("dataset/fulfillment", "epoch")
    hypervol_X_ = log.watch("bg/hypervolume_bg", "epoch")

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

        if epoch.val % config.update_pretrained_every_n_steps == 0 and config.scalarization == "improvement":
            with timer.section("update_base_model"):
                trainer.update_base_model()
        if epoch.val % config.sample_nm1_every_n_steps == 0 and config.scalarization == "improvement":
            with timer.section("sample_bg"):
                trainer.fix_optimization_problem()
                hypervol_X_.val = trainer.hypervolume_X_.median().item()
        if epoch.val % config.resample_every_n_steps == 0:
            with timer.section("generate_dataset"):
                trainer.update_exploration_model()
                dataset, advantages, fv = trainer.generate_dataset_fv()
                
                fulfillment.val = len(dataset) / config.num_samples
    
                if dataset:
                    first_var.val = torch.median(fv).item()     
                       
        if epoch.val % config.save_every_n_steps == 0:            
            with timer.section("save_checkpoint"):
                save_training_checkpoint(
                    run_resolution.run_dir,
                    next_epoch=epoch.val,
                    trainer_state=trainer.training_state_dict(),
                    loop_state={
                        "dataset": dataset,
                        "advantages": advantages,
                    },
                )

        if not dataset or advantages is None:
            print("No valid dataset or advantages, skipping epoch.")
            continue
        
        print(f"Epoch {epoch.val} starting with dataset size: {len(dataset)} and advantages size: {len(advantages)}")
        with timer.section("finetune"):
            timed_stats = trainer.finetune(dataset, advantages)
            
        print(f"Epoch {epoch.val} completed")
        
        fulltime_stats = {f"full/{k}": np.mean(v) for k, v in timed_stats.items()}
        log.log_dict(fulltime_stats, 'epoch')
        
        trainer.update_off_policy_model()
                    
        rows = timer.summary()
        
        if epoch.val % config.evaluate_every_n_steps == 0:
            with timer.section("evaluate_hypervolume"):           
                samples_eval = sample_x(vol_samples, trainer, discretization_steps=config.num_integration_steps)
                first_val_eval, n_hv.val, full_hv.val, rewards, valid_frac.val = evaluate(trainer, samples_eval, hv_computer, n=config.n)
                first_val_median.val = torch.median(first_val_eval).item()
                first_val_mean.val = torch.mean(first_val_eval).item()
                (qed.val, sa.val), (qed_td.val, sa_td.val), (qed_t3.val, sa_t3.val) = summarize_rewards(rewards)

        print("\n=== Timing summary (by total time) ===")
        for name, cnt, total, mean, p50, p95 in rows:
            print(f"{name:30s}  n={cnt:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  "
                f"p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")


    log.finish()
    mark_run_complete(run_resolution.run_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
