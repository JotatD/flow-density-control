import numpy as np
from diffusiongym import Sample
from diffusiongym.molecules.rewards.utils import graph_to_mols, is_not_fragmented, is_valid
from diffusiongym.molecules.types import DDGraph
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from vendi_score.vendi import score_K
from posebusters import PoseBusters

def diversity_metrics(samples: list[Sample[DDGraph]]) -> tuple[float, float, float, float, float, float, float, float]:
    sample = Sample.concat(samples).sample
    mols: list[Chem.Mol] = graph_to_mols(sample)
    

    valid_mols: list[Chem.Mol] = []
    for mol in mols:
        if is_valid(mol) and is_not_fragmented(mol):
            valid_mols.append(mol)
            
    validity_2d = len(valid_mols) / len(mols) 
    if len(valid_mols) <= 1:
        return validity_2d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 2d metrics

    morgan_fingerprints = [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in valid_mols
    ]

    tanimoto_similarity = np.eye(len(valid_mols), dtype=np.float64)

    pairs = 0
    similarity_sum = 0.0
    for i, fingerprint in enumerate(morgan_fingerprints):
        for j in range(i):
            similarity = DataStructs.TanimotoSimilarity(fingerprint, morgan_fingerprints[j])
            tanimoto_similarity[i, j] = similarity
            tanimoto_similarity[j, i] = similarity
            pairs += 1
            similarity_sum += similarity

    diversity_tanimoto = 1.0 - similarity_sum / pairs

    vendi_tanimoto = float(score_K(tanimoto_similarity))

    auc_coverage_tanimoto = auc(tanimoto_similarity)    

    buster = PoseBusters(config="mol")
    busting = buster.bust(valid_mols).all(axis=1).values  # ty: ignore[unresolved-attribute]
    conformer_valid_mols = [mol for mol, is_valid in zip(valid_mols, busting) if is_valid]

    validity_3d = len(conformer_valid_mols) / len(mols)
    
    if len(conformer_valid_mols) <= 1:
        return validity_2d, diversity_tanimoto, vendi_tanimoto, auc_coverage_tanimoto, validity_3d, 0.0, 0.0, 0.0


    usrcat_descriptors = np.asarray(
        [rdMolDescriptors.GetUSRCAT(mol) for mol in conformer_valid_mols],
        dtype=np.float64,
    ).reshape(len(conformer_valid_mols), 60)


    usrcat_similarity = np.eye(len(conformer_valid_mols), dtype=np.float64)

    pairs = 0
    similarity_sum = 0.0
    for i, descriptor in enumerate(usrcat_descriptors):
        for j in range(i):
            similarity = rdMolDescriptors.GetUSRScore(descriptor, usrcat_descriptors[j])
            usrcat_similarity[i, j] = similarity
            usrcat_similarity[j, i] = similarity
            pairs += 1
            similarity_sum += similarity
    
    diversity_usrcat = 1.0 - similarity_sum / pairs
            
    vendi_usrcat = float(score_K(usrcat_similarity))
    
    auc_coverage_usrcat = auc(usrcat_similarity)
    
    
    
    
    return validity_2d, diversity_tanimoto, vendi_tanimoto, auc_coverage_tanimoto, validity_3d, diversity_usrcat, vendi_usrcat, auc_coverage_usrcat, 




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
