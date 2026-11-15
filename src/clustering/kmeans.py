"""Phân cụm K-Means (sklearn.cluster.KMeans). Ma trận được chuẩn hóa L2 trước khi phân cụm."""
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src import config


def run_kmeans(X, k):
    """Trả về (nhãn cụm, tâm cụm, inertia). Tâm cụm nằm trong không gian đã chuẩn hóa L2."""
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(normalize(X))
    return km.labels_, km.cluster_centers_, km.inertia_
