import numpy as np
import daal4py as d4p
from statistics import mean
from typing import List


def kNN(matrix: np.ndarray, k: int) -> List[float]:
    training = d4p.kdtree_knn_classification_training(k=k)
    training = training.compute(matrix)
    prediction = d4p.kdtree_knn_classification_prediction()
    prediction = prediction.compute(matrix, training.model)
    distances = prediction.prediction
    print(distances)
    result = np.sort(list(mean(dist) for dist in distances))
    return result
