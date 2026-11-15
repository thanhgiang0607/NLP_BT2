"""Phân cụm K-Medoids (PAM) trên khoảng cách cosine, dùng thư viện `kmedoids`."""
import kmedoids
from sklearn.metrics.pairwise import cosine_distances

from src import config


def run_kmedoids(X, k):
    """Trả về (nhãn cụm, chỉ số bài làm tâm, inertia = tổng khoảng cách tới tâm)."""
    D = cosine_distances(X)
    km = kmedoids.KMedoids(k, metric="precomputed", method="pam",
                           random_state=config.RANDOM_STATE).fit(D)
    return km.labels_, km.medoid_indices_, km.inertia_
