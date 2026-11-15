"""Trích xuất top từ khóa TF-IDF và các bài viết gần tâm cụm nhất cho từng cụm.

Đầu vào:
    - Ma trận & vectorizer TF-IDF (models/tfidf_*) để tính trọng số từ khóa trung bình từng cụm.
    - Ma trận đặc trưng (bow_lsa / tfidf_lsa / embedding), nhãn cụm và tâm cụm (hoặc chỉ số medoid).

Đầu ra:
    - Top 10-15 từ khóa TF-IDF nổi bật của mỗi cụm (top_keywords).
    - 3-5 bài viết có độ tương đồng cosine lớn nhất với tâm cụm trong không gian chuẩn hóa L2
      (nearest_articles kèm tiêu đề + đoạn đầu ~120 từ để gửi Gemini).

Phụ trách: Xuân Hoàng
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from src.representation.bow_tfidf import top_terms


def top_keywords(X_tfidf, vectorizer, labels, n: int = 15) -> dict[int, list[str]]:
    """Lấy top-n từ khóa TF-IDF có trọng số trung bình cao nhất của mỗi cụm."""
    return {int(c): words for c, words in top_terms(X_tfidf, vectorizer, labels, n=n).items()}


def cluster_centers_l2(X: np.ndarray, centers_or_medoids: np.ndarray) -> np.ndarray:
    """Trả về tâm cụm đã chuẩn hóa L2 cho cả K-Means (ma trận 2D) và K-Medoids (chỉ số medoid 1D)."""
    Xn = normalize(X)
    arr = np.asarray(centers_or_medoids)
    # K-Medoids trả về mảng 1D chỉ số bài làm tâm -> lấy vector của các bài đó từ Xn;
    # K-Means trả về ma trận tâm 2D -> chuẩn hóa L2 để dùng chung phép đo cosine.
    return Xn[arr.astype(int)] if arr.ndim == 1 else normalize(arr)


def nearest_articles(
    X: np.ndarray,
    labels,
    centers_or_medoids: np.ndarray,
    df: pd.DataFrame,
    n: int = 5,
    snippet_words: int = 120,
) -> dict[int, list[dict]]:
    """Tìm n bài viết gần tâm cụm nhất cho từng cụm, kèm tiêu đề và đoạn đầu (~120 từ)."""
    Xn = normalize(X)
    C = cluster_centers_l2(X, centers_or_medoids)
    labels = np.asarray(labels)
    out: dict[int, list[dict]] = {}
    for cid in sorted(set(labels)):
        c = int(cid)
        idx = np.where(labels == cid)[0]
        # Hai vector đã chuẩn hóa L2 nên tích vô hướng chính là độ tương đồng cosine
        sims = Xn[idx] @ C[c]
        order = np.argsort(-sims)[:n]
        items = []
        for pos in order:
            row = df.iloc[idx[pos]]
            content = str(row["content"]) if "content" in df.columns and pd.notna(row["content"]) else ""
            # Chỉ lấy ~120 từ đầu (sapo + mở bài) để tiết kiệm token gửi Gemini mà vẫn đủ ý chính
            items.append({
                "id": str(row["id"]) if "id" in df.columns else str(idx[pos]),
                "title": str(row["title"]),
                "snippet": " ".join(content.split()[:snippet_words]),
                "similarity": round(float(sims[pos]), 4),
            })
        out[c] = items
    return out


def nearest_titles(
    X: np.ndarray,
    labels,
    centers_or_medoids: np.ndarray,
    titles,
    n: int = 5,
) -> dict[int, list[str]]:
    """Tìm n tiêu đề bài viết gần tâm/medoid cụm nhất cho từng cụm."""
    arts = nearest_articles(X, labels, centers_or_medoids, pd.DataFrame({"title": list(titles)}), n=n)
    return {c: [a["title"] for a in items] for c, items in arts.items()}
