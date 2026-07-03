from math import isfinite

import torch
from tqdm.auto import tqdm

from diffusiongym.environments import EndpointEnvironment
from diffusiongym.molecules.flowmol import FlowMolBaseModel

from genexp.mo import DXTBDipoleL2, DXTBEnergy
from genexp.trainers.rew_diff import RewDiff
from utils import resolve_config, seed_everything
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/reward_dxtb.yaml")
    parser.add_argument("--config_idx", type=int, default=None)
    parser.add_argument("--list_configs", action="store_true")
    parser.add_argument("--problem", choices=("energy", "dipole_l2"), default="energy")
    return parser.parse_args()


def build_environment(config, reward, device):
    base_model = FlowMolBaseModel(config.model_name, (1, 2, 2, 2), device)
    env = EndpointEnvironment(
        base_model,
        reward,
        discretization_steps=config.adjoint_matching.sampling.num_integration_steps,
    )
    sample = env.sample

    def sample_fixed_num_atoms(*args, **kwargs):
        kwargs.setdefault("n_atoms", int(config.fixed_num_atoms))
        return sample(*args, **kwargs)

    env.sample = sample_fixed_num_atoms
    return env

def evaluate_mean(trainer, num_samples: int):
    """Evaluate the mean reward of the fine model."""
    left = num_samples
    batch_size = trainer.config.batch_size
    rewards = []
    with torch.no_grad():
        while left > 0:
            samples = trainer.sample_trajectories().rewards
            rewards.append(samples)
            left -= batch_size 
    return torch.stack(rewards).mean().item()

def main():
    args = parse_args()
    config, _ = resolve_config(args)
    seed_everything(int(config.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_cls = {"energy": DXTBEnergy, "dipole_l2": DXTBDipoleL2}[args.problem]
    reward = reward_cls(fixed_num_atoms=int(config.fixed_num_atoms))
    env = build_environment(config, reward, device)
    trainer = RewDiff(config, env, device=device)

    num_eval_samples = int(config.get("num_eval_samples", 16))
    loaded_problem_mean = evaluate_mean(trainer, num_samples=num_eval_samples)
    
    print(
        f"problem={args.problem} problem_eval=loaded num_samples={num_eval_samples} "
        f"problem_mean={loaded_problem_mean:.6f}",
        flush=True,
    )

    global_step = 0
    for md_iteration in tqdm(range(config.num_md_iterations)):
        for adjoint_iteration in range(config.adjoint_matching.num_iterations):
            global_step += 1
            am_dataset = trainer.generate_dataset()
            loss = trainer.finetune(am_dataset, steps=None)

            problem_mean = evaluate_mean(trainer, num_samples=num_eval_samples)
            loss_text = "nan" if not isfinite(loss) else f"{loss:.6f}"
            print(
                f"md={md_iteration + 1} adjoint={adjoint_iteration + 1} "
                f"step={global_step} loss={loss_text} "
                f"problem_eval=inner num_samples={num_eval_samples} "
                f"problem_mean={problem_mean:.6f}",
                flush=True,
            )
            if loss_text == "nan":
                print("NaN loss encountered, stopping training.", flush=True)
                return
        trainer.update_base_model()


if __name__ == "__main__":
    main()
