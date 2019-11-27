import numpy as np
from multiprocessing import cpu_count
from n2 import HnswIndex
from typing import List


def kNN(matrix: np.ndarray, k: int) -> List[float]:
    index = HnswIndex(matrix.shape[1], 'L2')
    for sample in matrix:
        index.add_data(sample)
    index.build(m=32, max_m0=48, ef_construction=int(k * 1.1), n_threads=cpu_count())

    result = []
    for i in range(0, matrix.shape[0]):
        results = index.search_by_id(i, k, include_distances=True)
        result.append(np.mean(np.sqrt(np.array([dist for _, dist in results]))))
    return np.sort(result)
