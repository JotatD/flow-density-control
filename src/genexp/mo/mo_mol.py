from typing import Any

import torch
from diffusiongym.molecules import DDGraph
from diffusiongym.molecules.rewards.utils import graph_to_mols, is_not_fragmented, is_valid, safe_mmff_relax
from diffusiongym.molecules.rewards.xtb import parallel_xtb
from posebusters import PoseBusters
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    QED,
    Crippen,
)

# from src.utils import write_sdf_file
from genexp.mo.base import MOReward
from genexp.mo.sa_score import calculateScore


class MolecularMetrics(MOReward[DDGraph]):
    def __init__(self, do_relax: bool = True, full_bust: bool = True) -> None:
        RDLogger.DisableLog("rdApp.*")
        identity_fn = lambda x: x
        self.relax = safe_mmff_relax if do_relax else identity_fn
        self.buster = PoseBusters(config="mol" if full_bust else "mol_fast", max_workers=None)
        self.num_rew = 3
        self.ref_point = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        sa_n = (10 - sa) / 9
        return sa_n

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        mols = graph_to_mols(sample)

        valid_mols: list[Chem.Mol] = []
        valid_indices: list[int] = []
        for i, mol in enumerate(mols):
            if is_valid(mol) and is_not_fragmented(mol):
                try:
                    relaxed_mol = self.relax(mol)
                    relaxed_mol_is_busted = self.buster.bust(mol).all(axis=None)
                    if relaxed_mol_is_busted:
                        relaxed_mol = None
                except Exception:  # noqa: BLE001
                    relaxed_mol = None
                if relaxed_mol is not None:
                    valid_mols.append(relaxed_mol)
                    valid_indices.append(i)

        xtb_results = parallel_xtb(valid_mols) if valid_mols else []


        rewards = torch.zeros((len(sample), 3), device=sample.device, dtype=torch.float32)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)

        for idx in valid_indices:
            mol = mols[idx]
            res = xtb_results[valid_indices.index(idx)] if valid_mols else None
            if res is None:
                continue
            rewards[idx, 0] = float(self.calculate_qed(mol))
            rewards[idx, 1] = float(self.calculate_sa(mol))
            rewards[idx, 2] = float(res.dipole_moment) / 20.0
            valids[idx] = True

        return rewards, {"valids": valids}


class TopologyMetrics(MOReward[DDGraph]):
    def __init__(self, do_relax: bool = True, full_bust: bool = True) -> None:
        RDLogger.DisableLog("rdApp.*")
        identity_fn = lambda x: x
        self.relax = safe_mmff_relax if do_relax else identity_fn
        self.buster = PoseBusters(config="mol" if full_bust else "mol_fast", max_workers=None)
        self.num_rew = 2
        self.ref_point = torch.tensor([0.0, 0.0], dtype=torch.float32)

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        sa_n = (10 - sa) / 9
        return sa_n

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        mols = graph_to_mols(sample)

        valid_mols: list[Chem.Mol] = []
        valid_indices: list[int] = []
        for i, mol in enumerate(mols):
            if is_valid(mol) and is_not_fragmented(mol):
                try:
                    relaxed_mol = self.relax(mol)
                except Exception:  # noqa: BLE001
                    relaxed_mol = None
                if relaxed_mol is not None:
                    valid_mols.append(relaxed_mol)
                    valid_indices.append(i)

        rewards = torch.zeros((len(sample), self.num_rew), device=sample.device, dtype=torch.float32)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)

        if not valid_mols:
            return rewards, {"valids": valids}

        posebuster_valids = self.buster.bust(valid_mols).all(axis=1).values  # ty: ignore[unresolved-attribute]
        valid_indices = [idx for idx, is_valid in zip(valid_indices, posebuster_valids) if is_valid]

        for idx in valid_indices:
            mol = mols[idx]
            rewards[idx, 0] = float(self.calculate_qed(mol))
            rewards[idx, 1] = float(self.calculate_sa(mol))
            valids[idx] = True

        return rewards, {"valids": valids}


class RDkitReward(MOReward[DDGraph]):
    def __init__(self, rewards: list[str], valid_2d: str = "none", valid_3d: str = "none") -> None:
        RDLogger.DisableLog("rdApp.*")
        
        if valid_3d != "none":
            full_bust = "mol" if (valid_3d == "full") else "mol_fast"
            self.buster = PoseBusters(config=full_bust, max_workers=None)
        else:
            self.buster = None
            
        self.validate_2d: bool = (valid_2d == "full")
        
        self.rewards = rewards
        self.num_rew = len(rewards)
        self.ref_point = torch.zeros(self.num_rew, dtype=torch.float32)

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

        rewards = torch.zeros((len(sample), self.num_rew), device=sample.device, dtype=torch.float32)
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