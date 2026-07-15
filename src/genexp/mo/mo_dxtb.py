"""DXTB endpoint molecule rewards."""

from typing import Any, Callable, Sequence

import dgl
import dxtb
import torch

from diffusiongym.molecules.types import DDGraph

from genexp.mo.base import MOReward


ANGSTROM_TO_BOHR = 1.8897259886
GEOM_ATOM_TYPE_MAP = ("C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I")
ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
GEOM_CHARGE_OFFSET = -2.0
GEOM_MAX_CHARGE_CLASS = 5


def unbatch_dxtb_graphs(graph: dgl.DGLGraph) -> list[dgl.DGLGraph]:
    return dgl.unbatch(graph)


class _DXTBReward(MOReward[DDGraph]):
    """Base class for scalar endpoint molecule rewards computed with DXTB."""

    invalid_val = 0.0
    ref_point = torch.tensor([0.0], dtype=torch.float32)

    def __init__(self, fixed_num_atoms: int = 10, atom_type_map: Sequence[str] = GEOM_ATOM_TYPE_MAP, num_rew: int = 1) -> None:
        super().__init__(num_rew=num_rew, ref_point=self.ref_point)
        self.fixed_num_atoms = fixed_num_atoms
        self.atom_type_map = tuple(atom_type_map)

    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        graphs = unbatch_dxtb_graphs(sample.graph)
        reward_shape = (len(sample),) if self.num_rew == 1 else (len(sample), self.num_rew)
        rewards = torch.full(reward_shape, self.invalid_val, dtype=torch.float32, device=sample.device)
        valids = torch.zeros(len(sample), dtype=torch.bool, device=sample.device)
        reasons: list[str | None] = [None] * len(sample)

        for idx, graph in enumerate(graphs):
            try:
                value = self._evaluate_graph(graph)
                rewards[idx] = value.reshape(self.num_rew).to(device=sample.device, dtype=torch.float32)
                valids[idx] = True
            except Exception as exc:
                reasons[idx] = type(exc).__name__

        return rewards, {"valids": valids, "reasons": reasons}

    def objective(self, calc: Any, positions: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DXTBReward subclasses must implement objective")

    def _evaluate_graph(self, graph: dgl.DGLGraph) -> torch.Tensor:
        if graph.num_nodes() != self.fixed_num_atoms:
            raise ValueError(f"DXTBReward expects {self.fixed_num_atoms} atoms per molecule; got {graph.num_nodes()}")

        atom_type_idx = graph.ndata["a_t"].argmax(dim=-1)
        atom_type_idx = torch.where(atom_type_idx < len(self.atom_type_map), atom_type_idx, torch.zeros_like(atom_type_idx))
        atomic_numbers = torch.tensor([ATOMIC_NUMBERS[symbol] for symbol in self.atom_type_map], dtype=torch.long, device=graph.device)
        numbers = atomic_numbers[atom_type_idx.to(device=graph.device)]

        positions = graph.ndata["x_t"].to(dtype=torch.double) * ANGSTROM_TO_BOHR

        if "c_t" not in graph.ndata:
            charge = torch.zeros((), dtype=torch.double, device=graph.device)
        else:
            charge_idx = graph.ndata["c_t"].argmax(dim=-1)
            atom_charges = charge_idx.clamp(max=GEOM_MAX_CHARGE_CLASS).to(dtype=torch.double)
            charge = (atom_charges + GEOM_CHARGE_OFFSET).sum()

        dd = {"dtype": positions.dtype, "device": positions.device}
        field = torch.zeros(3, **dd, requires_grad=True)
        electric_field = dxtb.components.field.new_efield(field, **dd)
        calc = dxtb.calculators.GFN1Calculator(numbers, interaction=electric_field, **dd)
        return self.objective(calc, positions, charge)

    @staticmethod
    def _silent_dxtb_call(fn: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        with dxtb.OutputHandler.with_verbosity(0):
            return fn(*args, **kwargs)


class DXTBEnergy(_DXTBReward):
    """Lower physical energies are better, so this returns ``-energy``"""

    def objective(self, calc: Any, positions: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        energy = self._silent_dxtb_call(calc.get_energy, positions, chrg=charge, maxiter=500)
        return -energy


class DXTBDipoleL2(_DXTBReward):
    """DXTB dipole L2 norm reward."""

    @torch.enable_grad()
    def objective(self, calc: Any, positions: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        dipole = self._silent_dxtb_call(calc.get_dipole, positions, chrg=charge)
        return dipole.norm(dim=-1)


class DXTBTask(_DXTBReward):
    """DXTB two-objective reward with negative energy and dipole L2 norm."""
    ref_point = torch.tensor([0.0, 0.0], dtype=torch.float32)
    
    def __init__(self, fixed_num_atoms: int = 10, atom_type_map: Sequence[str] = GEOM_ATOM_TYPE_MAP) -> None:
        super().__init__(fixed_num_atoms=fixed_num_atoms, atom_type_map=atom_type_map, num_rew=2)

    @torch.enable_grad()
    def objective(self, calc: Any, positions: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        energy = self._silent_dxtb_call(calc.get_energy, positions, chrg=charge, maxiter=500).reshape(())
        dipole = self._silent_dxtb_call(calc.get_dipole, positions, chrg=charge).norm(dim=-1).reshape(())
        return torch.stack([-energy, dipole])
