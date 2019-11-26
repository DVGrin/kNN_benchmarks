# Benchmarking nearest neighbour libraries
Searching for nearest neighbours in high dimensional spaces is an important problem, and there no one silver bullet approach for every task.
This repository contains benchmarks of several Python libraries, and its goal is to compare various existing implementations of nearest neighbour search and select one that is most suited to our problem space.

There are several other benchmarking suites for comparing nearest neighbour search implementations on the Internet, most comprehensive being [Erik Bernhardsson's repository](https://github.com/erikbern/ann-benchmarks), but most of those test for slightly different usecases.

# Evaluated libraries
 - [Annoy](https://github.com/spotify/annoy)
 - [hnswlib](https://github.com/nmslib/hnsw)
 - [faiss](https://github.com/facebookresearch/faiss)
 - [scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html)
 - [n2](https://github.com/kakao/n2)   

# Context
We are benchmarking nearest neighbour methods as the part of a following problem - consider an error log for a distributed computing cluster. Those logs can regularly reach tens or even hundreds of thousands of entries, which makes human analysis of the logs pretty infeasible. 

We are using nearest neighbour search as a part of an algorithm for clusterising those logs to identify common problems and anomalous errors which may give us the ability to correct those problems and predict future difficulties.

# Dataset

A file with error logs from ATLAS experimant at CERN is used for those benchmarks. For each launch we choose a random sample of logs and use it for every algorithm. We use different neighbour numbers, both constant for different sample sizes and scaling with those (as square root of number of samples). Full results can be seen in [img](https://github.com/DVGrin/kNN_benchmarks/tree/master/img) folder of that repository.

Word2Vec algorithm is used to convert strings of errors into vectors which are then used for nearest neighbour searching. Note that due to a nature of a problem we have to find nearest neighbours for every point in the dataset, we don't have the distinction between the train dataset and test dataset.

# Results

![img](./img/kNN_benchmarks_sqrt_neighbours.png)
![img](./img/kNN_benchmark_40000_samples_200_neighbours.png)

As we can see for now, hnswlib should be the best choice for us from the tested libraries, winning pretty consistently on every dataset size.
