import argparse
import json
import math
import pickle
from argparse import Namespace
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.flowmol import GEOMBaseModel
from diffusiongym.molecules.rewards.utils import graph_to_mols
from tqdm.auto import tqdm
from utils import seed_everything

from genexp.mo.mo_mol import RDkitReward
from genexp.mo.moses import diversity_metrics_2d, diversity_metrics_3d
from genexp.mo.utils import HVComputer
from genexp.resume import mark_run_complete, resolve_run
from genexp.trainers.diff_blend import BlendEnvironment
from genexp.trainers.utils import StepTimer
from genexp.wandb_log import WandbLogger
from hv_nft import evaluate, jack_hv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--force_new_start", action="store_true")
    parser.add_argument("--project_name", type=str, default="retro_compatibility")
    parser.add_argument("--run_name", type=str, default="diffusion_blend")
    parser.add_argument("--seed", type=int, default=5)

    # algorithm
    parser.add_argument("--sampling_mode", type=str, required=True, choices=["full", "proba"])
    parser.add_argument("--tradeoff_sampling", type=str, default="random", choices=["random", "sobol", "das_dennis"])

    # checkpoint cadence and integration
    parser.add_argument("--epochs", type=int, default=170)
    parser.add_argument("--evaluate_every_n_steps", type=int, default=10)
    parser.add_argument("--num_integration_steps", type=int, default=100)

    # evaluation size
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--vol_samples", type=int, default=320)
    parser.add_argument("--samples_per_tradeoff", type=int, default=8)
    parser.add_argument("--evaluation_samples", type=int, default=None, help="Randomly retain this many valid samples for reward and diversity metrics; raw outputs remain complete")
    parser.add_argument("--jackknife_permutations", type=int, default=100_000)
    parser.add_argument("--num_diversity_samples", type=int, default=1000, help="Deprecated; diversity uses the reward-evaluation samples")

    # validity
    parser.add_argument("--validate_2d", type=str, default="none", choices=["none", "full"])
    parser.add_argument("--validate_3d", type=str, default="none", choices=["none", "fast", "full"])

    # reward-specific model runs
    parser.add_argument("--qed_folder", type=str, default=None)
    parser.add_argument("--sa_folder", type=str, default=None)
    parser.add_argument("--dipole_folder", type=str, default=None)

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


def das_dennis_simplex_tradeoffs(num_samples: int, num_rews: int) -> list[list[float]]:
    """Return a complete simplex lattice, requiring an exact lattice point count."""
    if num_samples <= 0 or num_rews < 2:
        raise ValueError("Das-Dennis requires positive num_samples and at least two rewards")
    partitions = 1
    while math.comb(partitions + num_rews - 1, num_rews - 1) < num_samples:
        partitions += 1
    count = math.comb(partitions + num_rews - 1, num_rews - 1)
    if count != num_samples:
        lower_count = math.comb(partitions + num_rews - 2, num_rews - 1) if partitions > 1 else 0
        raise ValueError(
            f"Das-Dennis cannot generate exactly {num_samples} weights for {num_rews} rewards. "
            f"Choose vol_samples / samples_per_tradeoff equal to a complete lattice size "
            f"(nearby sizes: {lower_count or count}, {count})."
        )

    def lattice(remaining: int, dimensions: int, prefix: list[float]):
        if dimensions == 1:
            yield prefix + [remaining / partitions]
        else:
            for value in range(remaining + 1):
                yield from lattice(remaining - value, dimensions - 1, prefix + [value / partitions])

    return list(lattice(partitions, num_rews, []))


def generate_tradeoffs(sampling: str, num_samples: int, num_rews: int, seed: int) -> list[list[float]]:
    if sampling == "random":
        return uniform_simplex_tradeoffs(num_samples, num_rews, seed)
    if sampling == "sobol":
        return sobol_simplex_tradeoffs(num_samples, num_rews, seed)
    if sampling == "das_dennis":
        return das_dennis_simplex_tradeoffs(num_samples, num_rews)
    raise ValueError(f"Unknown tradeoff sampling method: {sampling}")


def _read_run_configuration(folder: Path) -> dict:
    manifest_path = folder / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    return manifest["configuration"]


def configured_model_runs(config: Namespace) -> list[tuple[str, Path]]:
    model_runs = [
        (reward_name, Path(folder))
        for reward_name, folder in (("qed", config.qed_folder), ("sa", config.sa_folder), ("dipole", config.dipole_folder))
        if folder is not None
    ]
    if len(model_runs) not in {2, 3}:
        raise ValueError("Provide model folders for exactly two or three of: qed, sa, dipole")
    return model_runs


def validate_model_runs(model_runs: list[tuple[str, Path]]) -> float:
    run_configs = []
    for reward_name, folder in model_runs:
        run_config = _read_run_configuration(folder)
        if run_config.get("reward") != reward_name:
            raise ValueError(f"Expected a {reward_name} model run in {folder}")
        run_configs.append(run_config)

    source_kl_weight = float(run_configs[0]["alpha"])
    if any(not math.isclose(float(run_config["alpha"]), source_kl_weight) for run_config in run_configs[1:]):
        raise ValueError("All reward models must use the same KL weight (alpha)")
    return source_kl_weight


def resolve_checkpoints(model_runs: list[tuple[str, Path]], last_epoch: int, cadence: int) -> list[tuple[int, list[Path]]]:
    if last_epoch < 0:
        raise ValueError("epochs must be nonnegative")
    if cadence <= 0:
        raise ValueError("evaluate_every_n_steps must be positive")
    epochs = list(range(0, last_epoch + 1, cadence))
    checkpoints = [(epoch, [folder / f"model_epoch_{epoch}.pt" for _, folder in model_runs]) for epoch in epochs]
    missing = [str(path) for _, paths in checkpoints for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing expected model checkpoints:\n" + "\n".join(missing))
    return checkpoints


def load_model_state(model: GEOMBaseModel, path: Path, device: torch.device) -> None:
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()


def sample_rewards(
    env: BlendEnvironment,
    reward: RDkitReward,
    tradeoffs: list[list[float]],
    samples_per_tradeoff: int,
    discretization_steps: int,
    batch_size: int,
    fixed_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor, list[DDGraph]]:
    all_rewards = []
    all_valids = []
    sampled_graphs = []
    sample_kwargs = {"n_atoms": fixed_atoms} if fixed_atoms > 0 else {}
    with torch.no_grad():
        for blend_vector in tradeoffs:
            num_valids = 0
            attempts = 0
            max_attempts = 10 * samples_per_tradeoff
            while num_valids < samples_per_tradeoff and attempts < max_attempts:
                remaining = samples_per_tradeoff - num_valids
                batch = min(batch_size, max_attempts - attempts, remaining * 5)
                samples = env.sample(batch, blend_vector=blend_vector, discretization_steps=discretization_steps, pbar=True, **sample_kwargs)
                batch_rewards, info = reward(samples.sample, samples.latent)
                batch_rewards = batch_rewards.detach().cpu()
                batch_valids = info["valids"].detach().cpu()

                if batch_valids.sum().item() >= remaining:
                    batch = int(torch.nonzero(batch_valids, as_tuple=False)[remaining - 1].item()) + 1

                all_rewards.append(batch_rewards[:batch])
                all_valids.append(batch_valids[:batch])
                sampled_graphs.extend(sample.sample for sample in islice(samples, batch))
                num_valids += int(batch_valids[:batch].sum().item())
                attempts += batch

            if num_valids < samples_per_tradeoff:
                print(f"Warning: tradeoff {blend_vector} produced {num_valids}/{samples_per_tradeoff} valid molecules after {attempts} attempts")

    return torch.cat(all_rewards, dim=0), torch.cat(all_valids, dim=0), sampled_graphs


def select_evaluation_indices(validity: torch.Tensor, num_samples: int | None, seed: int) -> torch.Tensor:
    indices = torch.nonzero(validity, as_tuple=True)[0]
    if num_samples is None:
        return indices
    if num_samples <= 0:
        raise ValueError("evaluation_samples must be positive")
    if len(indices) < num_samples:
        print(f"Warning: only {len(indices)}/{num_samples} valid samples available for evaluation")
        return indices
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randperm(len(indices), generator=generator)[:num_samples]
    return indices[selected].sort().values


def main(config: Namespace) -> None:
    assert config.vol_samples > 0 and config.samples_per_tradeoff > 0, "vol_samples and samples_per_tradeoff must be positive"
    assert config.vol_samples % config.samples_per_tradeoff == 0, "vol_samples must be divisible by samples_per_tradeoff"
    assert config.n > 0, "n must be positive"
    if config.evaluation_samples is not None and not 0 < config.evaluation_samples <= config.vol_samples:
        raise ValueError("evaluation_samples must be positive and no greater than vol_samples")

    model_runs = configured_model_runs(config)
    reward_names = [reward_name for reward_name, _ in model_runs]
    source_kl_weight = validate_model_runs(model_runs)
    checkpoints = resolve_checkpoints(model_runs, config.epochs, config.evaluate_every_n_steps)
    num_tradeoffs = config.vol_samples // config.samples_per_tradeoff
    tradeoffs = generate_tradeoffs(config.tradeoff_sampling, num_tradeoffs, len(model_runs), config.seed)
    config.reward_names = reward_names
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
    c_wandb = vars(config)
    c_wandb.update({"scalarization": "blend"})
    log = WandbLogger(
        project_name=config.project_name,
        config=c_wandb,
        use_wandb=config.wandb,
        run_name=run_resolution.run_dir.name,
        id=run_resolution.wandb_run_id,
        resume="allow",
        dir=str(run_resolution.run_dir),
    )
    epoch_metric = log.set_step_metric(checkpoints[0][0], "epoch")

    valid_frac = log.watch("valid_fraction", "epoch")

    diversity_tanimoto = log.watch("diversity/diversity_tanimoto", "epoch")
    vendi_tanimoto = log.watch("diversity/vendi_tanimoto", "epoch")
    auc_tanimoto = log.watch("diversity/auc_coverage_tanimoto", "epoch")
    diversity_usrcat = log.watch("diversity/diversity_usrcat", "epoch")
    vendi_usrcat = log.watch("diversity/vendi_usrcat", "epoch")
    auc_usrcat = log.watch("diversity/auc_coverage_usrcat", "epoch")

    obj_dir = run_resolution.run_dir / "objs"
    scores_dir = run_resolution.run_dir / "scores"
    obj_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer = StepTimer(device=device)
    reward_config = {"valid_3d": config.validate_3d, "valid_2d": config.validate_2d, "invalid_val": config.invalid_val}
    reward = RDkitReward(reward_names, **reward_config)
    hv_computer = HVComputer(ref_point=reward.ref_point, num_rew=reward.num_rew)

    models = [GEOMBaseModel(device=device) for _ in model_runs]
    for model in models:
        model.requires_grad_(False)
    env = BlendEnvironment(models, discretization_steps=config.num_integration_steps, sampling_mode=config.sampling_mode)

    for checkpoint_epoch, checkpoint_paths in tqdm(checkpoints, desc="Checkpoint epochs"):
        if checkpoint_epoch == 0:
            continue
        epoch_metric.val = checkpoint_epoch
        seed_everything(config.seed)
        with timer.section("load_state"):
            for model, checkpoint_path in zip(models, checkpoint_paths, strict=True):
                load_model_state(model, checkpoint_path, device)
        with timer.section("evaluate_hypervolume"):
            rewards, validity, reward_samples = sample_rewards(
                env=env,
                reward=reward,
                tradeoffs=tradeoffs,
                samples_per_tradeoff=config.samples_per_tradeoff,
                discretization_steps=config.num_integration_steps,
                batch_size=config.batch_size,
                fixed_atoms=config.fixed_A,
            )
            mols = list(graph_to_mols(DDGraph.collate(reward_samples)))
            with (obj_dir / f"obj_{checkpoint_epoch}.pkl").open("wb") as file:
                pickle.dump(mols, file)

            evaluation_indices = select_evaluation_indices(validity, config.evaluation_samples, config.seed)
            np.savez(
                scores_dir / f"score_{checkpoint_epoch}.npz",
                scores=rewards.numpy(),
                validity=validity.numpy(),
                evaluation_indices=evaluation_indices.numpy(),
            )
            valid_rewards = rewards[evaluation_indices]
            score_metrics = evaluate(valid_rewards, reward.reward_names, hv_computer, n=config.n)
            for jackknife_n in (100, 500):
                if valid_rewards.shape[0] > jackknife_n:
                    with timer.section(f"jackknife_{jackknife_n}"):
                        jack_mean, jack_low, jack_high, jack_variance = jack_hv(
                            valid_rewards,
                            hv_computer,
                            n=jackknife_n,
                            permutations=config.jackknife_permutations,
                            ci=0.95,
                        )
                    score_metrics.update({
                        f"jack/mean_{jackknife_n}": jack_mean,
                        f"jack/low_{jackknife_n}": jack_low,
                        f"jack/high_{jackknife_n}": jack_high,
                        f"jack/var_{jackknife_n}": jack_variance,
                    })
            log.log_dict(score_metrics, "epoch")

            valid_frac.val = validity.float().mean().item()
            clean_molecules = [mols[index] for index in evaluation_indices.tolist()]
            _, diversity_tanimoto.val, vendi_tanimoto.val, auc_tanimoto.val = diversity_metrics_2d(clean_molecules)
            _, diversity_usrcat.val, vendi_usrcat.val, auc_usrcat.val = diversity_metrics_3d(
                clean_molecules, full_bust=config.validate_3d == "full"
            )

        print("\n=== Timing summary (by total time) ===")
        for name, count, total, mean, p50, p95 in timer.summary():
            print(f"{name:30s}  n={count:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")

    log.finish()
    mark_run_complete(run_resolution.run_dir)


if __name__ == "__main__":
    main(parse_args())
