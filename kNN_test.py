import re
import math
import time

from random import choices
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import cpu_count

import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from nltk.tokenize import TreebankWordTokenizer

import knn_methods


def _open_file(filename: str, sample_size: int) -> List[str]:
    with open(filename) as log_file:
        file_contents = log_file.readlines()
        if filename[-4:] == ".csv":
            file_contents = file_contents[1:]  # We don't need column names
        file_contents = list(map(lambda x: x.strip(), file_contents))
    return choices(file_contents, k=sample_size)


def _clean_file_contents(string_list: List[str]) -> List[str]:
    # TODO: Check UIDs and UUIDs
    # TODO: Add other substitutions
    # _uid = r'[0-9a-zA-Z]{12,128}'
    # _uuid = r'[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89aAbB][a-f0-9]{3}-[a-f0-9]{12}'
    _line_number = r'(at line[:]*\s*\d+)'
    _file_or_directory = r"No such file or directory: '.*'"

    for string in string_list:
        if not string:
            continue
        string = string.split(',')[2]  # ! This assumes that error message is in the third column
        string = ' '.join(string.split())  # Clear duplicate whitespace
        string = re.sub(_line_number, "at line LINE_NUMBER", string)
        string = re.sub(_file_or_directory, "No such file or directory: 'PATH'", string)
        # string = re.sub(_uid, "UID", string)
        # string = re.sub(_uuid, "UUID", string)

    return string_list


def _tokenize_strings(errors_list: List[str]) -> List[List[str]]:
    tokenized = []
    for line in errors_list:
        tokenized.append(TreebankWordTokenizer().tokenize(line))
    return tokenized


def _vectorize_strings(error_tokens_list: List[List[str]]) -> np.ndarray:
    # TODO: Clear the code
    # TODO: Check Word2Vec parameters
    # TODO: model[token] leads to DeprecationWarning
    result = []
    model = Word2Vec(error_tokens_list, size=100, window=5, min_count=1, workers=cpu_count(), iter=10)
    for error in error_tokens_list:
        sentence_vector: List[Any] = []
        token_number = 0
        for token in error:
            if token_number == 0:
                sentence_vector = model[token]
            else:
                sentence_vector = np.add(sentence_vector, model[token])
            token_number += 1

        result.append(np.asarray(sentence_vector) / token_number)
    return np.array(result)


def prepare_file(filename: str, sample_size: int) -> np.ndarray:
    strings = _open_file(filename, sample_size)
    strings = _clean_file_contents(strings)
    tokens = _tokenize_strings(strings)
    tokens = _vectorize_strings(tokens)
    return tokens


def benchmark_kNN_methods(matrix: np.ndarray, k: Optional[int], n_runs: int) -> Dict[str, Tuple[float, float]]:
    # TODO: compare kNN results to be close
    methods = knn_methods.method_list
    if k is None:
        k = int(math.sqrt(matrix.shape[0]))

    results: Dict[str, List[float]] = {}
    for name, method in methods.items():
        results[name] = []
        for i in range(n_runs):
            start_time = time.perf_counter()
            distances = method(matrix, k)
            results[name].append(time.perf_counter() - start_time)
        print(f"{name}: {distances[-5:]}")

    avg_results: Dict[str, Tuple[float, float]] = {}
    for name in results:
        avg_results[name] = (np.mean(results[name]), np.std(results[name]))
    return avg_results


def plot_benchmark_results_bar(benchmark_times: Dict[str, Tuple[float, float]], sample_size: int, k: Optional[int]) -> None:
    if k is None:
        k = int(math.sqrt(sample_size))
    plt.figure()
    means = [value[0] for value in benchmark_times.values()]
    plt.bar(benchmark_times.keys(), means)
    plt.ylabel("Execution time")
    plt.suptitle(f"kNN method execution time for {sample_size} samples, {k} neighbours")
    plt.savefig(f"./img/kNN_benchmark_{sample_size}_samples_{k}_neighbours.png")


def plot_benchmark_results(results: Dict[int, Dict[str, Tuple[float, float]]], k: Optional[int]) -> None:
    plt.figure()
    sample_sizes = list(results.keys())
    benchmark_times: Dict[str, List[float]] = dict(zip(
        list(results.values())[0],
        (list() for i in range(len(results)))
    ))
    benchmark_std: Dict[str, List[float]] = dict(zip(
        list(results.values())[0],
        (list() for i in range(len(results)))
    ))
    for method_results in results.values():
        for method, elapsed_time in method_results.items():
            benchmark_times[method].append(elapsed_time[0])
            benchmark_std[method].append(elapsed_time[1])
    for method, times in benchmark_times.items():
        plt.errorbar(sample_sizes, times, yerr=benchmark_std[method], fmt="-o", label=method)

    plt.xlabel("Sample size")
    plt.ylabel("Execution time (s)")
    plt.legend()
    if k is None:
        plt.suptitle(r"kNN method execution time for $\sqrt{k}$ neighbours")
        plt.savefig(f"./img/kNN_benchmarks_sqrt_neighbours.png")
    else:
        plt.suptitle(f"kNN method execution time for {k} neighbours")
        plt.savefig(f"./img/kNN_benchmarks_{k}_neighbours.png")


def run_benchmarks(filename: str, sample_sizes: List[int], k: Optional[int] = None, n_runs: int = 1) -> Dict[int, Dict[str, Tuple[float, float]]]:
    results = {}
    for sample_size in sample_sizes:
        data = prepare_file(filename, sample_size)
        benchmark_times = benchmark_kNN_methods(data, k=k, n_runs=n_runs)
        results[sample_size] = benchmark_times
        plot_benchmark_results_bar(benchmark_times, sample_size, k)
    plot_benchmark_results(results, k)
    return results


if __name__ == "__main__":
    results = run_benchmarks("../datasets/error_logs_test.csv", [1000, 5000, 10000, 20000, 40000], k=None, n_runs=5)
    for sample_size, benchmark_times in results.items():
        print(f"{sample_size} samples:")
        for algo_name, algo_time in benchmark_times.items():
            print(f"\t{algo_name}: {algo_time[0]:.4} ± {algo_time[1]:.4} s")
