import numpy as np
import annoy
from statistics import mean
from typing import List


def kNN(matrix: np.ndarray, k: int) -> List[float]:
    index = annoy.AnnoyIndex(matrix.shape[1], "euclidean")
    for i, item in enumerate(matrix):
        index.add_item(i, matrix[i])
    index.build(20)
    distances = []
    for i in range(matrix.shape[0]):
        distances.append(index.get_nns_by_item(i, k, include_distances=True)[1])
    result = np.sort(list(mean(dist) for dist in distances))
    return result
