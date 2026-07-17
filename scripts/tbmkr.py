import numpy as np
from genexp.mo.utils import plot_objective_points
import torch
import matplotlib.pyplot as plt

from genexp.mo.mo_dxtb import DXTBTask
from diffusiongym.molecules.rewards.xtb import XTBTask
from diffusiongym import EndpointEnvironment
from diffusiongym.molecules.flowmol import GEOMBaseModel, get_upper_edge_mask
reward = DXTBTask(fixed_num_atoms=10)
import pickle as pkl
base_model = GEOMBaseModel(device="cuda")


def main():
    sample_path = '/home/juan.guevara/Code/flow-density-control/assets/dxtb_10A/data/dec_u.pkl'

    z_data = []

    z_info = []
    with open(sample_path, 'rb') as f:
        samples = pkl.load(f)

    for i, s in enumerate(samples):
        g = s.clone()
        for key in list(g.graph.ndata.keys()):
            if key.endswith("_t"):
                g.graph.ndata[key[:-2] + "_1"] = g.graph.ndata.pop(key)

        for key in list(g.graph.edata.keys()):
            if key.endswith("_t"):
                g.graph.edata[key[:-2] + "_1"] = g.graph.edata.pop(key)

        #To enable usage with SampledMolecule from flowmol
        g.graph.edata["ue_mask"] = get_upper_edge_mask(g.graph)

        dxtb_reward = DXTBTask(fixed_num_atoms=10)
        # xtb_reward = XTBTask(do_relax=True)

        dxtb_value, dxtb_info = dxtb_reward(s, s)
        # xtb_value, xtb_info = xtb_reward(g, g)

        print("DXTB:", dxtb_value[0], dxtb_info)

        z_data.append(dxtb_value[0].detach().cpu().numpy())
        
        z_info.append(dxtb_info) 
        print('-'*30)
        
    final_info = z_info[0]
    for k, v in final_info.items():
        if isinstance(v, list):
            for inf in z_info[1:]:
                final_info[k].extend(inf[k])
        if isinstance(v, np.ndarray):
            final_info[k] = np.concatenate([inf[k] for inf in z_info], axis=0)
        if isinstance(v, torch.Tensor):
            final_info[k] = np.concatenate([inf[k].detach().cpu().numpy() for inf in z_info], axis=0)


    np.save("assets/dxtb_10A/data/obj_f.npy", z_data)
    with open("assets/dxtb_10A/data/info_f.pkl", "wb") as f:
        pkl.dump(final_info, f)
        

def inspect():
    rewards = np.load("assets/dxtb_10A/data/obj_f.npy")
    infos = pkl.load(open("assets/dxtb_10A/data/info_f.pkl", "rb"))
    ax = plot_objective_points(torch.tensor(rewards), None)
    # save fig
    ax.figure.savefig("obj_f.png", dpi=300)
    
    print("rewards shape:", rewards.shape, rewards.mean(axis=0), rewards.std(axis=0))
    print("infos keys:", infos.keys(), {k: len(v) for k, v in infos.items()})
    
def main():
    u_data = np.load("assets/dxtb_10A/data/obj_f.npy")
    q05, q95 = np.quantile(u_data[:, 0], [0.002, 0.996])
    data = u_data[(u_data[:, 0] >= q05) & (u_data[:, 0] <= q95)]

    f05, f95 = np.quantile(u_data[:, 1], [0.000, 0.996])
    data = u_data[(u_data[:, 1] >= f05) & (u_data[:, 1] <= f95)]

    random_points = np.random.uniform(low=[q05, f05], high=[q95, f95], size=(1000, 2))
    ax = plot_objective_points(ambient=torch.tensor(data), special=torch.tensor(random_points))

    ax.figure.savefig("obj_f_random.png", dpi=300)
    
    np.save("assets/dxtb_10A/data/obj.npy", data)

if __name__ == "__main__":
    main()
    inspect()