import numpy as np
from statistics import mean
from sklearn.neighbors import NearestNeighbors
from typing import List


def kNN(matrix: np.ndarray, k: int) -> List[float]:
    neigh = NearestNeighbors(n_neighbors=k, metric='l2')
    nbrs = neigh.fit(matrix)
    distances, indices = nbrs.kneighbors(matrix)
    avg_distances = []
    for line in distances:
        avg_distances.append(mean(line))
    return np.sort(avg_distances)
