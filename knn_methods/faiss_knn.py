import numpy as np
import faiss
from statistics import mean
from typing import List


def kNN(matrix: np.ndarray, k: int) -> List[float]:
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    distances, _ = index.search(matrix, k)
    result = np.sort(list(mean(np.sqrt(dist)) for dist in distances))
    return result
