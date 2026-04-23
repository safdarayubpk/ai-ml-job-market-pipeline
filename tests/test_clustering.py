import numpy as np
from scipy.sparse import csr_matrix
from ml.clustering import cluster_jobs, cluster_summary, find_optimal_k


def _make_matrix(n_samples: int = 40, n_features: int = 50) -> csr_matrix:
    rng = np.random.RandomState(42)
    return csr_matrix(rng.rand(n_samples, n_features))


def test_cluster_jobs_returns_label_array():
    matrix = _make_matrix(40, 50)
    _, labels = cluster_jobs(matrix, k=3)
    assert len(labels) == 40


def test_cluster_jobs_label_range():
    matrix = _make_matrix(40, 50)
    _, labels = cluster_jobs(matrix, k=4)
    assert set(labels).issubset({0, 1, 2, 3})


def test_cluster_summary_counts_sum_to_total():
    labels = np.array([0, 0, 1, 1, 2, 2, 2])
    summary = cluster_summary(labels)
    assert sum(v["count"] for v in summary.values()) == 7


def test_cluster_summary_percentages_sum_to_100():
    labels = np.array([0, 0, 1, 1, 2])
    summary = cluster_summary(labels)
    total_pct = sum(v["pct"] for v in summary.values())
    assert abs(total_pct - 100.0) < 0.5


def test_find_optimal_k_returns_int_in_range():
    matrix = _make_matrix(40, 50)
    k = find_optimal_k(matrix, k_range=range(2, 5))
    assert isinstance(k, int)
    assert 2 <= k <= 4
