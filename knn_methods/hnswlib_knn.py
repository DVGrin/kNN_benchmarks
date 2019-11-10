import numpy as np
import hnswlib
from statistics import mean
from typing import List


def kNN(matrix: np.ndarray, k: int) -> List[float]:
    index = hnswlib.Index(space='l2', dim=matrix.shape[1])
    index.init_index(max_elements=matrix.shape[0], ef_construction=int(k * 1.1), M=48)
    index.add_items(matrix)
    labels, distances = index.knn_query(matrix, k=k)
    result = np.sort(list(mean(np.sqrt(dist)) for dist in distances))
    return result
