"""
Author: William Carlsson

How to run:
- Execute the script using `python task1.py`.
"""

import numpy as np
from itertools import combinations
from collections import defaultdict

def read_pdb(filepath):
    """Parse PDB file to extract alpha-carbon coordinates for each model."""
    models = defaultdict(list)
    current_model = None
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("MODEL"):
                current_model = int(line.split()[1])
            elif line.startswith("ATOM") and line[13:15].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                models[current_model].append((x, y, z))
            elif line.startswith("ENDMDL"):
                current_model = None
    return {k: np.array(v) for k, v in models.items()}

def compute_internal_distances(model_coords):
    """Compute pairwise distances between alpha-carbon atoms for a given model."""
    num_atoms = len(model_coords)
    distances = np.zeros((num_atoms, num_atoms))
    for i, j in combinations(range(num_atoms), 2):
        dist = np.linalg.norm(model_coords[i] - model_coords[j])
        distances[i, j] = dist
        distances[j, i] = dist

    return distances

def compute_sse(model1, model2):
    """Compute the sum of squared errors (SSE) between two models."""
    return np.sum((model1 - model2) ** 2)

def cluster_models(pdb_file):
    """Cluster models based on SSE and identify max/min pairs."""
    models = read_pdb(pdb_file)
    distances = {k: compute_internal_distances(v) for k, v in models.items()}
    
    sse_scores = {}
    for (model1, dist1), (model2, dist2) in combinations(distances.items(), 2):
        sse_scores[(model1, model2)] = compute_sse(dist1, dist2)
    
    max_pair = max(sse_scores, key=sse_scores.get)
    min_pair = min(sse_scores, key=sse_scores.get)
    
    return {
        "max_pair": max_pair,
        "max_sse": sse_scores[max_pair],
        "min_pair": min_pair,
        "min_sse": sse_scores[min_pair],
        "sse_scores": sse_scores,
    }

pdb_file = "4hir.pdb"
results = cluster_models(pdb_file)

print("Max SSE Pair:", results["max_pair"], "Value:", results["max_sse"])
print("Min SSE Pair:", results["min_pair"], "Value:", results["min_sse"])
