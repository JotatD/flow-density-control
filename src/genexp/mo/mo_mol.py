from typing import Any

import torch
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.rewards.utils import graph_to_mols, is_not_fragmented, is_valid
from posebusters import PoseBusters
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

# from src.utils import write_sdf_file
from genexp.mo.base import MOReward
from genexp.mo.sa_score import calculateScore


class TopologyMetrics(MOReward[DDGraph]):
    def __init__(self, valid_2d: str = "none", valid_3d: str = "none", invalid_val: float = 0.0) -> None:
        RDLogger.DisableLog("rdApp.*")

        if valid_3d != "none":
            full_bust = "mol" if (valid_3d == "full") else "mol_fast"
            self.buster = PoseBusters(config=full_bust, max_workers=None)
        else:
            self.buster = None

        self.validate_2d: bool = valid_2d == "full"
        self.invalid_val = invalid_val
        
        if self.invalid_val > 0:
            raise ValueError("invalid_val must be <= 0 for TopologyMetrics")

        self.num_rew = 2
        self.ref_point = torch.full((self.num_rew,), fill_value=self.invalid_val, dtype=torch.float32)

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        sa_n = (10 - sa) / 9
        return sa_n


    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        mols = graph_to_mols(sample)
        valid_mols: list[Chem.Mol] = []
        valid_indices: list[int] = []

        for i, mol in enumerate(mols):
            if not self.validate_2d or (is_valid(mol) and is_not_fragmented(mol)):
                valid_mols.append(mol)
                valid_indices.append(i)

        rewards = torch.full((len(sample), self.num_rew), fill_value=self.invalid_val, device=sample.device, dtype=torch.float32)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)

        if not valid_mols:
            return rewards, {"valids": valids}

        if self.buster is not None:
            posebuster_valids = self.buster.bust(valid_mols).all(axis=1).values  # ty: ignore[unresolved-attribute]
        else:
            posebuster_valids = [True] * len(valid_mols)
            
        valid_indices = [idx for idx, is_valid in zip(valid_indices, posebuster_valids) if is_valid]

        for idx in valid_indices:
            mol = mols[idx]
            rewards[idx, 0] = float(self.calculate_qed(mol))
            rewards[idx, 1] = float(self.calculate_sa(mol))
            valids[idx] = True

        return rewards, {"valids": valids}


class RDkitReward(MOReward[DDGraph]):
    def __init__(self, rewards: list[str], valid_2d: str = "none", valid_3d: str = "none", invalid_val: float = 0.0) -> None:
        RDLogger.DisableLog("rdApp.*")
        
        if valid_3d != "none":
            full_bust = "mol" if (valid_3d == "full") else "mol_fast"
            self.buster = PoseBusters(config=full_bust, max_workers=None)
        else:
            self.buster = None
            
        self.validate_2d: bool = (valid_2d == "full")
        if invalid_val > 0:
            raise ValueError("invalid_val must be <= 0 for RDkitReward")

        self.rewards = rewards
        self.num_rew = len(rewards)
        self.invalid_val = invalid_val
        self.ref_point = torch.full((self.num_rew,), fill_value=self.invalid_val, dtype=torch.float32)

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        sa_n = (10 - sa) / 9
        return sa_n
    
    def evaluate_reward(self, mol: Chem.Mol) -> torch.Tensor:
        reward_values = []
        for reward_name in self.rewards:
            if reward_name == "qed":
                reward_values.append(self.calculate_qed(mol))
            elif reward_name == "sa":
                reward_values.append(self.calculate_sa(mol))
            else:
                raise ValueError(f"Unknown reward: {reward_name}")
        return torch.tensor(reward_values, dtype=torch.float32)

    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        mols = graph_to_mols(sample)

        valid_mols: list[Chem.Mol] = []
        valid_indices: list[int] = []
        
        for i, mol in enumerate(mols):
            if not self.validate_2d or (is_valid(mol) and is_not_fragmented(mol)):
                valid_mols.append(mol)
                valid_indices.append(i)

        rewards = torch.full((len(sample), self.num_rew), fill_value=self.invalid_val, device=sample.device, dtype=torch.float32)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)

        if not valid_mols:
            return rewards, {"valids": valids}

        if self.buster is not None:
            posebuster_valids = self.buster.bust(valid_mols).all(axis=1).values  # ty: ignore[unresolved-attribute]
        else:
            posebuster_valids = [True] * len(valid_mols)
            
        valid_indices = [idx for idx, is_valid in zip(valid_indices, posebuster_valids) if is_valid]

        for idx in valid_indices:
            mol = mols[idx]
            rewards[idx] = self.evaluate_reward(mol)
            valids[idx] = True

        return rewards, {"valids": valids}