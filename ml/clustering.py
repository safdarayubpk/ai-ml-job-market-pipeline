import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.sparse import spmatrix


def find_optimal_k(matrix: spmatrix, k_range: range = range(3, 9)) -> int:
    """
    Choose k using silhouette score. Higher is better.
    Uses sample_size=min(500, n_samples) to keep it fast on large datasets.
    """
    n_samples = matrix.shape[0]
    if n_samples < max(k_range):
        k_range = range(2, max(3, n_samples // 2))

    scores: dict[int, float] = {}
    for k in k_range:
        if k >= n_samples:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        sample_size = min(500, n_samples)
        scores[k] = silhouette_score(matrix, labels, sample_size=sample_size, random_state=42)

    return max(scores, key=scores.get) if scores else 3


def cluster_jobs(matrix: spmatrix, k: int) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans with given k. Returns (fitted model, label array)."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)
    return km, labels


def save_pca_plot(
    matrix: spmatrix, labels: np.ndarray, output_path: str = "cluster_plot.png"
) -> None:
    """Reduce to 2D with PCA and save a scatter plot to output_path."""
    dense = matrix.toarray()
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(dense)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=60)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("AI/ML Job Listing Clusters (PCA 2D Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def cluster_summary(labels: np.ndarray) -> dict[int, dict]:
    """Return count and percentage for each cluster ID."""
    total = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    return {
        int(k): {"count": int(c), "pct": round(c / total * 100, 1)}
        for k, c in zip(unique, counts)
    }
