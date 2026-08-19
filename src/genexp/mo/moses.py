import numpy as np
from diffusiongym import Sample
from diffusiongym.molecules.rewards.utils import graph_to_mols, is_not_fragmented, is_valid
from diffusiongym.molecules.types import DDGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from vendi_score.vendi import score_K


def diversity_metrics(samples: list[Sample[DDGraph]]) -> tuple[float, float, float, float]:
    sample = Sample.concat(samples).sample
    mols: list[Chem.Mol] = graph_to_mols(sample)

    valid_mols: list[Chem.Mol] = []
    for mol in mols:
        if is_valid(mol) and is_not_fragmented(mol):
            valid_mols.append(mol)
            
    validity = len(valid_mols) / len(mols) 
    if len(valid_mols) <= 1:
        return validity, 0.0, 0.0, 0.0

    usrcat_descriptors = np.asarray(
        [rdMolDescriptors.GetUSRCAT(mol) for mol in valid_mols],
        dtype=np.float64,
    ).reshape(len(valid_mols), 60)


    usrcat_similarity = np.eye(len(valid_mols), dtype=np.float64)

    pairs = 0
    similarity_sum = 0.0
    for i, descriptor in enumerate(usrcat_descriptors):
        for j in range(i):
            similarity = rdMolDescriptors.GetUSRScore(descriptor, usrcat_descriptors[j])
            usrcat_similarity[i, j] = similarity
            usrcat_similarity[j, i] = similarity
            pairs += 1
            similarity_sum += similarity
    
    diversity = 1.0 - similarity_sum / pairs
            
    vendi_usrcat = float(score_K(usrcat_similarity))
    
    auc_coverage = auc(usrcat_similarity)
    
    return validity, diversity, vendi_usrcat, auc_coverage


def num_threshold_clusters(S, threshold=0.85):
    S = np.asarray(S)

    centers = []

    for i in range(len(S)):
        if not centers:
            centers.append(i)
            continue

        # Candidate must be dissimilar enough from ALL existing centers
        if np.max(S[i, centers]) < threshold:
            centers.append(i)

    return len(centers)


def auc(S):
    points = 100
    thresholds = [i / points for i in range(points + 1)]
    y = [num_threshold_clusters(S, threshold=t) for t in thresholds]
    auc = np.trapz(y, thresholds)

    return auc