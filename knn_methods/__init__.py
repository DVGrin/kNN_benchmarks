from . import n2_knn
from . import hnswlib_knn
from . import sklearn_knn
from . import annoy_knn
# from . import faiss_knn

method_list = {
    "n2": n2_knn.kNN,
    "hnswlib": hnswlib_knn.kNN,
    "sklearn": sklearn_knn.kNN,
    "annoy": annoy_knn.kNN,
    # "faiss": faiss_knn.kNN  # ! Only works in conda at the moment. Results between annoy and hnswlib
}
