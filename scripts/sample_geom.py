

import os

import numpy as np
import pkl
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
        discretization_steps=40,
    )
    max_batch = 64
    all_rewards = []
    all_samples = []
    all_info = []
    try: 
        while num_samples > 0:
            batch_size = min(num_samples, max_batch)
            samples = env.sample(batch_size, fixed_num_atoms=10)
            
            all_samples.extend(samples.samples)
            all_rewards.append(samples.rewards.numpy())
            all_info.append(samples.info)
            num_samples -= batch_size
    finally:
        os.makedirs("../assets/dxtb", exist_ok=True)
        all_samples = np.concatenate(all_samples, axis=0)
        np.save("../assets/dxtb/rewards_10A.npy", all_samples)
        
        with open("../assets/dxtb/samples_10A.pkl", "wb") as f:
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
        
        with open("../assets/dxtb/info_10A.pkl", "wb") as f:
            pkl.dump(final_info, f)

def inspect():
    rewards = np.load("../assets/dxtb/rewards_10A.npy")
    infos = pkl.load(open("../assets/dxtb/info_10A.pkl", "rb"))
    samples = pkl.load(open("../assets/dxtb/samples_10A.pkl", "rb"))
    
    print("rewards", rewards.shape, rewards.mean(), rewards.std())
    print("samples", len(samples), samples[:10])
    print("info", {k: v.shape if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor) else len(v) for k, v in infos.items()})
    
    
if __name__ == "__main__":
    collect()
    inspect()