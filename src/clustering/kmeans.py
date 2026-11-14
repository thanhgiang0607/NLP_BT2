"""Phân cụm K-Means (sklearn.cluster.KMeans). Ma trận được chuẩn hóa L2 trước khi phân cụm."""
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src import config


def run_kmeans(X, k, random_state=config.RANDOM_STATE):
    """Trả về (nhãn cụm, model đã fit)."""
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(normalize(X))
    return labels, model
