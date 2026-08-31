import argparse
import json
import math
from argparse import Namespace
from math import ceil
from pathlib import Path

import torch
from diffusiongym import Sample
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.flowmol import GEOMBaseModel
from diffusiongym.molecules.rewards.utils import graph_to_mols
from tqdm.auto import tqdm
from utils import seed_everything

from genexp.mo.mo_mol import TopologyMetrics
from genexp.mo.moses import diversity_metrics_2d
# from genexp.mo.moses import diversity_metrics_3d
from genexp.mo.utils import HVComputer
from genexp.resume import mark_run_complete, resolve_run
from genexp.trainers.diff_blend import BlendEnvironment
from genexp.trainers.utils import StepTimer
from genexp.wandb_log import WandbLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--force_new_start", action="store_true")
    parser.add_argument("--project_name", type=str, default="blend_2")
    parser.add_argument("--run_name", type=str, default="diffusion_blend")
    parser.add_argument("--seed", type=int, default=5)

    # algorithm
    parser.add_argument("--sampling_mode", type=str, required=True, choices=["full", "proba"])
    parser.add_argument("--tradeoff_sampling", type=str, default="random", choices=["random", "sobol"])

    # checkpoint cadence and integration
    parser.add_argument("--epochs", type=int, default=170)
    parser.add_argument("--evaluate_every_n_steps", type=int, default=10)
    parser.add_argument("--num_integration_steps", type=int, default=100)

    # evaluation size
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--vol_samples", type=int, default=320)
    parser.add_argument("--samples_per_tradeoff", type=int, default=8)
    parser.add_argument("--num_diversity_samples", type=int, default=1000)

    # validity
    parser.add_argument("--validate_2d", type=str, default="none", choices=["none", "full"])
    parser.add_argument("--validate_3d", type=str, default="none", choices=["none", "fast", "full"])

    # reward-specific model runs
    parser.add_argument("--qed_folder", type=str, default="output/qed/hv_nft_20260830_125045_050407")
    parser.add_argument("--sa_folder", type=str, default="output/sa/hv_nft_20260830_141511_942475")

    # molecule and reward modifiers
    parser.add_argument("--fixed_A", type=int, default=10)
    parser.add_argument("--invalid_val", type=float, default=-1.0)
    return parser.parse_args()


def uniform_simplex_tradeoffs(num_samples: int, num_rews: int, seed: int) -> list[list[float]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    uniforms = torch.rand(num_samples, num_rews, generator=generator).clamp_min(torch.finfo(torch.float32).tiny)
    exponentials = -torch.log(uniforms)
    return (exponentials / exponentials.sum(dim=1, keepdim=True)).tolist()


def sobol_simplex_tradeoffs(num_samples: int, num_rews: int, seed: int) -> list[list[float]]:
    if num_rews == 1:
        return [[1.0] for _ in range(num_samples)]
    points = torch.quasirandom.SobolEngine(dimension=num_rews - 1, scramble=True, seed=seed).draw(num_samples)
    points = points.sort(dim=1).values
    boundaries = torch.cat([torch.zeros(num_samples, 1), points, torch.ones(num_samples, 1)], dim=1)
    return torch.diff(boundaries, dim=1).tolist()


def generate_tradeoffs(sampling: str, num_samples: int, num_rews: int, seed: int) -> list[list[float]]:
    if sampling == "random":
        return uniform_simplex_tradeoffs(num_samples, num_rews, seed)
    return sobol_simplex_tradeoffs(num_samples, num_rews, seed)


def _read_run_configuration(folder: Path) -> dict:
    manifest_path = folder / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    return manifest["configuration"]


def validate_model_runs(qed_folder: Path, sa_folder: Path, fixed_atoms: int) -> float:
    qed_config = _read_run_configuration(qed_folder)
    sa_config = _read_run_configuration(sa_folder)
    if qed_config.get("reward") != "qed":
        raise ValueError(f"Expected a QED model run in {qed_folder}")
    if sa_config.get("reward") != "sa":
        raise ValueError(f"Expected an SA model run in {sa_folder}")
    if not math.isclose(float(qed_config["alpha"]), float(sa_config["alpha"])):
        raise ValueError("QED and SA models must use the same KL weight (alpha)")
    if int(qed_config["fixed_A"]) != int(sa_config["fixed_A"]) or int(qed_config["fixed_A"]) != fixed_atoms:
        raise ValueError("QED and SA training runs and blend inference must use the same fixed_A")
    return float(qed_config["alpha"])


def resolve_checkpoint_pairs(qed_folder: Path, sa_folder: Path, last_epoch: int, cadence: int) -> list[tuple[int, Path, Path]]:
    if last_epoch < 0:
        raise ValueError("epochs must be nonnegative")
    if cadence <= 0:
        raise ValueError("evaluate_every_n_steps must be positive")
    epochs = list(range(0, last_epoch + 1, cadence))
    pairs = [(epoch, qed_folder / f"model_epoch_{epoch}.pt", sa_folder / f"model_epoch_{epoch}.pt") for epoch in epochs]
    missing = [str(path) for _, qed_path, sa_path in pairs for path in (qed_path, sa_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing expected model checkpoints:\n" + "\n".join(missing))
    return pairs


def load_model_state(model: GEOMBaseModel, path: Path, device: torch.device) -> None:
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()


def sample_rewards(
    env: BlendEnvironment,
    reward: TopologyMetrics,
    tradeoffs: list[list[float]],
    samples_per_tradeoff: int,
    discretization_steps: int,
    batch_size: int,
    fixed_atoms: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    results = []
    sample_kwargs = {"n_atoms": fixed_atoms} if fixed_atoms > 0 else {}
    with torch.no_grad():
        for blend_vector in tradeoffs:
            rewards = []
            valids = []
            left = samples_per_tradeoff
            while left > 0:
                batch = min(left, batch_size)
                samples = env.sample(batch, blend_vector=blend_vector, discretization_steps=discretization_steps, pbar=True, **sample_kwargs)
                batch_rewards, info = reward(samples.sample, samples.latent)
                rewards.append(batch_rewards.detach().cpu())
                valids.append(info["valids"].detach().cpu())
                left -= batch
            results.append((torch.cat(rewards, dim=0), torch.cat(valids, dim=0)))
    return results


def sample_x(
    env: BlendEnvironment,
    tradeoffs: list[list[float]],
    samples_per_tradeoff: int,
    num_samples: int,
    discretization_steps: int,
    batch_size: int,
    fixed_atoms: int,
) -> list[Sample[DDGraph]]:
    samples = []
    sample_kwargs = {"n_atoms": fixed_atoms} if fixed_atoms > 0 else {}
    with torch.no_grad():
        for blend_vector in tradeoffs:
            left = min(samples_per_tradeoff, num_samples - len(samples))
            while left > 0:
                batch = min(left, batch_size)
                sampled = env.sample(batch, blend_vector=blend_vector, discretization_steps=discretization_steps, pbar=True, **sample_kwargs)
                samples.extend([sample for sample in sampled])
                left -= batch
            if len(samples) == num_samples:
                break
    return samples


def compute_hypervolume(rewards: torch.Tensor, hv_computer: HVComputer) -> float:
    if rewards.shape[0] == 0:
        return 0.0
    return hv_computer(rewards.unsqueeze(0)).item()


def summarize_rewards(rewards: torch.Tensor) -> tuple[list[float], list[float], list[float]]:
    if rewards.shape[0] == 0:
        missing = [float("nan")] * rewards.shape[1]
        return missing, missing.copy(), missing.copy()
    top_decile_count = ceil(rewards.shape[0] * 0.1)
    means = rewards.mean(dim=0)
    top_decile_means = rewards.topk(top_decile_count, dim=0).values.mean(dim=0)
    top_3_means = rewards.topk(min(3, rewards.shape[0]), dim=0).values.mean(dim=0)
    return tuple(values.tolist() for values in (means, top_decile_means, top_3_means))


def evaluate_reward_results(
    results: list[tuple[torch.Tensor, torch.Tensor]],
    hv_computer: HVComputer,
    group_size: int,
    shuffle_seed: int,
) -> tuple[float, float, torch.Tensor, float]:
    rewards = torch.cat([preference_rewards for preference_rewards, _ in results], dim=0)
    valids = torch.cat([preference_valids for _, preference_valids in results], dim=0)
    valid_rewards = rewards[valids]
    full_hypervolume = compute_hypervolume(valid_rewards, hv_computer)

    generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)
    shuffled_rewards = valid_rewards[torch.randperm(valid_rewards.shape[0], generator=generator)]
    grouped_count = shuffled_rewards.shape[0] - shuffled_rewards.shape[0] % group_size
    if grouped_count > 0:
        grouped_rewards = shuffled_rewards[:grouped_count].reshape(-1, group_size, rewards.shape[1])
        n_hypervolume = hv_computer(grouped_rewards).mean().item()
    else:
        n_hypervolume = 0.0

    return n_hypervolume, full_hypervolume, valid_rewards, valids.float().mean().item()


def main(config: Namespace) -> None:
    assert config.vol_samples > 0 and config.samples_per_tradeoff > 0, "vol_samples and samples_per_tradeoff must be positive"
    assert config.vol_samples % config.samples_per_tradeoff == 0, "vol_samples must be divisible by samples_per_tradeoff"
    assert config.n > 0, "n must be positive"
    assert config.validate_2d == "none" or config.num_diversity_samples > 0, "num_diversity_samples must be positive when 2D validation is enabled"

    qed_folder = Path(config.qed_folder)
    sa_folder = Path(config.sa_folder)
    source_kl_weight = validate_model_runs(qed_folder, sa_folder, config.fixed_A)
    checkpoint_pairs = resolve_checkpoint_pairs(qed_folder, sa_folder, config.epochs, config.evaluate_every_n_steps)
    num_tradeoffs = config.vol_samples // config.samples_per_tradeoff
    tradeoffs = generate_tradeoffs(config.tradeoff_sampling, num_tradeoffs, 2, config.seed)
    num_diversity_tradeoffs = ceil(config.num_diversity_samples / config.samples_per_tradeoff) if config.validate_2d != "none" else 0
    diversity_tradeoffs = generate_tradeoffs(config.tradeoff_sampling, num_diversity_tradeoffs, 2, config.seed + 1)
    config.tradeoffs = tradeoffs
    config.source_kl_weight = source_kl_weight

    results_root = Path("output") / config.project_name
    run_resolution = resolve_run(config, results_root, config.run_name)
    if run_resolution.completed:
        print(f"Matching run is already complete: {run_resolution.run_dir}")
        return

    print(f"run_dir={run_resolution.run_dir}")
    print(f"sampling_mode={config.sampling_mode}")
    print(f"tradeoffs={tradeoffs}")
    log = WandbLogger(
        project_name=config.project_name,
        config=vars(config),
        use_wandb=config.wandb,
        run_name=run_resolution.run_dir.name,
        id=run_resolution.wandb_run_id,
        resume="allow",
        dir=str(run_resolution.run_dir),
    )
    epoch_metric = log.set_step_metric(checkpoint_pairs[0][0], "epoch")

    n_hv = log.watch("n_hypervolume", "epoch")
    full_hv = log.watch("full_hypervolume", "epoch")
    qed = log.watch("qed", "epoch")
    qed_td = log.watch("top_decile/qed", "epoch")
    qed_t3 = log.watch("top_3/qed", "epoch")
    sa = log.watch("sa", "epoch")
    sa_td = log.watch("top_decile/sa", "epoch")
    sa_t3 = log.watch("top_3/sa", "epoch")
    valid_frac = log.watch("valid_fraction", "epoch")

    # if config.validate_2d != "none":
    #     valid_2d = log.watch("diversity/validity_2d", "epoch")
    #     diversity_tanimoto = log.watch("diversity/diversity_tanimoto", "epoch")
    #     vendi_tanimoto = log.watch("diversity/vendi_tanimoto", "epoch")
    #     auc_tanimoto = log.watch("diversity/auc_coverage_tanimoto", "epoch")

    # if config.validate_3d != "none":
    #     valid_3d = log.watch("diversity/validity_3d", "epoch")
    #     diversity_usrcat = log.watch("diversity/diversity_usrcat", "epoch")
    #     vendi_usrcat = log.watch("diversity/vendi_usrcat", "epoch")
    #     auc_usrcat = log.watch("diversity/auc_coverage_usrcat", "epoch")

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer = StepTimer(device=device)
    reward_config = {"valid_3d": config.validate_3d, "valid_2d": config.validate_2d, "invalid_val": config.invalid_val}
    reward = TopologyMetrics(**reward_config)
    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)

    qed_model = GEOMBaseModel(device=device)
    sa_model = GEOMBaseModel(device=device)
    qed_model.requires_grad_(False)
    sa_model.requires_grad_(False)
    env = BlendEnvironment([qed_model, sa_model], discretization_steps=config.num_integration_steps, sampling_mode=config.sampling_mode)

    for checkpoint_epoch, qed_path, sa_path in tqdm(checkpoint_pairs, desc="Checkpoint epochs"):
        epoch_metric.val = checkpoint_epoch
        seed_everything(config.seed)
        with timer.section("load_state"):
            load_model_state(qed_model, qed_path, device)
            load_model_state(sa_model, sa_path, device)
        with timer.section("evaluate_hypervolume"):
            reward_results = sample_rewards(
                env=env,
                reward=reward,
                tradeoffs=tradeoffs,
                samples_per_tradeoff=config.samples_per_tradeoff,
                discretization_steps=config.num_integration_steps,
                batch_size=config.batch_size,
                fixed_atoms=config.fixed_A,
            )
            n_hv.val, full_hv.val, rewards, valid_frac.val = evaluate_reward_results(reward_results, hv_computer, group_size=config.n, shuffle_seed=config.seed)
            (qed.val, sa.val), (qed_td.val, sa_td.val), (qed_t3.val, sa_t3.val) = summarize_rewards(rewards)

        # if config.validate_2d != "none":
        #     with timer.section("evaluate_diversity"):
        #         samples_diversity = sample_x(
        #             env=env,
        #             tradeoffs=diversity_tradeoffs,
        #             samples_per_tradeoff=config.samples_per_tradeoff,
        #             num_samples=config.num_diversity_samples,
        #             discretization_steps=config.num_integration_steps,
        #             batch_size=config.batch_size,
        #             fixed_atoms=config.fixed_A,
        #         )
        #         sample = DDGraph.collate([sample.sample for sample in samples_diversity])
        #         mols = graph_to_mols(sample)
        #         valid_2d_count, diversity_tanimoto.val, vendi_tanimoto.val, auc_tanimoto.val = diversity_metrics_2d(mols)
        #         valid_2d.val = valid_2d_count / len(mols)

        # if config.validate_3d != "none":
        #     with timer.section("evaluate_diversity_3d"):
        #         samples_diversity = sample_x(
        #             env=env,
        #             tradeoffs=diversity_tradeoffs,
        #             samples_per_tradeoff=config.samples_per_tradeoff,
        #             num_samples=config.num_diversity_samples,
        #             discretization_steps=config.num_integration_steps,
        #             batch_size=config.batch_size,
        #             fixed_atoms=config.fixed_A,
        #         )
        #         sample = Sample.concat(samples_diversity).sample
        #         mols = graph_to_mols(sample)
        #         full_bust = config.validate_3d == "full"
        #         valid_3d_count, diversity_usrcat.val, vendi_usrcat.val, auc_usrcat.val = diversity_metrics_3d(mols, full_bust=full_bust)
        #         valid_3d.val = valid_3d_count / len(mols)

        print("\n=== Timing summary (by total time) ===")
        for name, count, total, mean, p50, p95 in timer.summary():
            print(f"{name:30s}  n={count:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")

    log.finish()
    mark_run_complete(run_resolution.run_dir)


if __name__ == "__main__":
    main(parse_args())
