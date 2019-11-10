from . import n2_knn
from . import hnswlib_knn
from . import sklearn_knn

method_list = {
    "n2": n2_knn.kNN,
    "hnswlib": hnswlib_knn.kNN,
    "sklearn": sklearn_knn.kNN
}
