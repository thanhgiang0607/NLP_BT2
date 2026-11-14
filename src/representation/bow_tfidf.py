"""Ma trận Bag-of-Words (CountVectorizer) và TF-IDF (TfidfVectorizer, ngram 1-2).

Đầu vào: cột `tokens` của data/processed/articles_clean.csv (đã tách từ, bỏ stopwords).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src import config
from src.representation.tokens import split_tokens


def load_corpus(path: Path | str = config.ARTICLES_CLEAN_CSV) -> tuple[pd.DataFrame, list[str]]:
    """Đọc articles_clean.csv, trả về (bảng dữ liệu, danh sách chuỗi tokens của từng bài)."""
    df = pd.read_csv(path, encoding="utf-8")
    df["tokens"] = df["tokens"].fillna("")
    return df, df["tokens"].tolist()


def build_bow(docs, min_df=config.MIN_DF, max_df=config.MAX_DF, ngram_range=(1, 1)):
    """Ma trận đếm từ (BoW). Trả về (ma trận, vectorizer đã học từ vựng)."""
    vec = CountVectorizer(tokenizer=split_tokens, token_pattern=None, lowercase=False,
                          min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    return vec.fit_transform(docs), vec


def build_tfidf(docs, min_df=config.MIN_DF, max_df=config.MAX_DF, ngram_range=config.NGRAM_RANGE):
    """Ma trận TF-IDF. Trả về (ma trận, vectorizer đã học từ vựng)."""
    vec = TfidfVectorizer(tokenizer=split_tokens, token_pattern=None, lowercase=False,
                          min_df=min_df, max_df=max_df, ngram_range=ngram_range, sublinear_tf=True)
    return vec.fit_transform(docs), vec


def save_matrix(name: str, X, vectorizer, models_dir: Path = config.MODELS_DIR) -> None:
    """Lưu ma trận (.npz) và vectorizer (.joblib) vào thư mục models/."""
    models_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(models_dir / f"{name}_matrix.npz", X.tocsr())
    joblib.dump(vectorizer, models_dir / f"{name}_vectorizer.joblib")


def load_matrix(name: str, models_dir: Path = config.MODELS_DIR):
    """Nạp lại (ma trận, vectorizer) đã lưu bằng save_matrix."""
    X = sp.load_npz(models_dir / f"{name}_matrix.npz")
    return X, joblib.load(models_dir / f"{name}_vectorizer.joblib")


def top_terms(X, vectorizer, labels, n=10) -> dict:
    """Top-n từ có trọng số trung bình cao nhất của từng chủ đề."""
    terms = np.array(vectorizer.get_feature_names_out())
    labels = np.asarray(labels)
    out = {}
    for lab in sorted(set(labels)):
        mean = np.asarray(X[labels == lab].mean(axis=0)).ravel()
        out[lab] = terms[np.argsort(mean)[::-1][:n]].tolist()
    return out


def matrix_stats(X) -> dict:
    """Thống kê nhanh: số bài, số từ, độ thưa của ma trận."""
    n_docs, n_terms = X.shape
    density = X.nnz / (n_docs * n_terms)
    return {"so_bai": n_docs, "so_tu": n_terms, "nnz": int(X.nnz),
            "do_thua_%": round(100 * (1 - density), 2)}


def build_and_save_all() -> dict:
    """Chạy toàn bộ: đọc articles_clean.csv -> BoW + TF-IDF -> lưu vào models/."""
    df, docs = load_corpus()
    X_bow, v_bow = build_bow(docs)
    X_tfidf, v_tfidf = build_tfidf(docs)
    save_matrix("bow", X_bow, v_bow)
    save_matrix("tfidf", X_tfidf, v_tfidf)
    return {"bow": matrix_stats(X_bow), "tfidf": matrix_stats(X_tfidf)}


if __name__ == "__main__":
    import json
    print(json.dumps(build_and_save_all(), ensure_ascii=False, indent=2))