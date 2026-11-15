"""Chỉ số: Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI, Purity."""
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import normalize


def contingency(true_labels, cluster_labels) -> pd.DataFrame:
    """Bảng chéo: mỗi hàng là một cụm, mỗi cột là một chủ đề thật, ô là số bài."""
    return pd.crosstab(pd.Series(cluster_labels, name="Cụm"),
                       pd.Series(true_labels, name="Chủ đề thật"))


def purity(true_labels, cluster_labels) -> float:
    """Purity = số bài thuộc chủ đề chiếm nhiều nhất của mỗi cụm, cộng lại, chia tổng số bài."""
    ct = contingency(true_labels, cluster_labels)
    return float(ct.max(axis=1).sum() / ct.values.sum())


def evaluate(X, labels, true_labels) -> dict:
    """Tính 6 chỉ số. Chỉ số nội tại tính trên X đã chuẩn hóa L2 (Silhouette dùng cosine)
    để K-Means và K-Medoids được so trên cùng một thước đo."""
    Xn = normalize(X)
    return {
        "silhouette": silhouette_score(Xn, labels, metric="cosine"),
        "davies_bouldin": davies_bouldin_score(Xn, labels),
        "calinski_harabasz": calinski_harabasz_score(Xn, labels),
        "ari": adjusted_rand_score(true_labels, labels),
        "nmi": normalized_mutual_info_score(true_labels, labels),
        "purity": purity(true_labels, labels),
    }
