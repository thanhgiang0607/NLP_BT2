"""Chọn k bằng Elbow + Silhouette."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score

from src import config
from src.clustering.kmeans import run_kmeans
from src.clustering.kmedoids import run_kmedoids
from src.visualization.plots import _save

MODELS = {"kmeans": run_kmeans, "kmedoids": run_kmedoids}


def scan_k(X, model, k_range=config.K_RANGE) -> pd.DataFrame:
    """Chạy mô hình với từng k, ghi lại inertia (Elbow) và Silhouette (cosine)."""
    rows = []
    for k in k_range:
        labels, _, inertia = MODELS[model](X, k)
        rows.append({"k": k, "inertia": inertia,
                     "silhouette": silhouette_score(X, labels, metric="cosine")})
    return pd.DataFrame(rows)


def best_k(scan: pd.DataFrame) -> int:
    """k có Silhouette cao nhất."""
    return int(scan.loc[scan["silhouette"].idxmax(), "k"])


def plot_elbow_silhouette(scan: pd.DataFrame, title="", name=None):
    """Vẽ Elbow (inertia) và Silhouette theo k, đánh dấu k tốt nhất."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(scan["k"], scan["inertia"], "o-")
    ax1.set(title="Elbow", xlabel="k", ylabel="Inertia")
    ax2.plot(scan["k"], scan["silhouette"], "o-", color="#D55E00")
    ax2.axvline(best_k(scan), ls="--", color="#999999")
    ax2.set(title="Silhouette", xlabel="k", ylabel="Silhouette (cosine)")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, name)
