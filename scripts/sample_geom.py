

import os

import numpy as np
import pickle as pkl
import torch
from diffusiongym.environments import EndpointEnvironment
from diffusiongym.molecules.flowmol import GEOMBaseModel

from genexp.mo import DXTBTask


def collect() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = GEOMBaseModel(device=device)
    reward = DXTBTask(fixed_num_atoms=10)
    num_samples = 10_000
    env = EndpointEnvironment(
        base_model,
        reward,
        discretization_steps=250,
    )
    max_batch = 128
    all_rewards = []
    all_samples = []
    all_info = []
    try: 
        while num_samples > 0:
            print("Sampling...")
            batch_size = min(num_samples, max_batch)
            samples = env.sample(batch_size, n_atoms=10, pbar=True)
            
            all_samples.extend(samples.sample)
            all_rewards.append(samples.rewards.numpy())
            all_info.append(samples.info)
            num_samples -= batch_size
            print(num_samples, "samples remaining")
    finally:
        os.makedirs("assets/dxtb_10A", exist_ok=True)
        all_rewards = np.concatenate(all_rewards, axis=0)
        np.save("assets/dxtb_10A/obj.npy", all_rewards)
        
        with open("assets/dxtb_10A/dec.pkl", "wb") as f:
            pkl.dump(all_samples, f)
            
        final_info = all_info[0]
        for k, v in final_info.items():
            if isinstance(v, list):
                for inf in all_info[1:]:
                    final_info[k].extend(inf[k])
            if isinstance(v, np.ndarray):
                final_info[k] = np.concatenate([inf[k] for inf in all_info], axis=0)
            if isinstance(v, torch.Tensor):
                final_info[k] = np.concatenate([inf[k].numpy() for inf in all_info], axis=0)
        
        with open("assets/dxtb_10A/info.pkl", "wb") as f:
            pkl.dump(final_info, f)

def inspect():
    rewards = np.load("assets/dxtb_10A/obj.npy")
    infos = pkl.load(open("assets/dxtb_10A/info.pkl", "rb"))
    samples = pkl.load(open("assets/dxtb_10A/dec.pkl", "rb"))
    print("rewards", rewards.shape, rewards.mean(axis=0), rewards.std(axis=0))
    print("samples", len(samples), samples[:10])
    print("info", {k: v.shape if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor) else len(v) for k, v in infos.items()})
    
    
if __name__ == "__main__":
    collect()
    inspect()