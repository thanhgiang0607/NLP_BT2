"""Phân cụm K-Medoids, metric='cosine'.

Dùng thư viện `kmedoids` (FasterPAM) vì `sklearn_extra` không chạy được với NumPy 2.
"""
import kmedoids

from src import config


def run_kmedoids(X, k, random_state=config.RANDOM_STATE):
    """Trả về (nhãn cụm, model đã fit). Chỉ số bài làm tâm cụm: model.medoid_indices_."""
    model = kmedoids.KMedoids(n_clusters=k, metric="cosine", method="fasterpam",
                              random_state=random_state)
    return model.fit(X).labels_, model
