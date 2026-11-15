"""Trích xuất đặc trưng LSA (TruncatedSVD) cho BoW và TF-IDF, đồng thời cung cấp hàm load_features().

Phụ trách: Gia Khang
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src import config
from src.representation.bow_tfidf import load_matrix, build_and_save_all as build_bow_tfidf
from src.representation.embedding import build_and_save_embedding


def fit_lsa(
    X_sparse,
    n_components: int = config.LSA_COMPONENTS,
    random_state: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, TruncatedSVD]:
    """Giảm chiều ma trận thưa bằng TruncatedSVD, trả về (ma trận dày, model SVD)."""
    # Đảm bảo n_components không vượt quá min(n_samples, n_features) - 1
    max_comp = min(X_sparse.shape) - 1
    k = min(n_components, max_comp)
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    X_reduced = svd.fit_transform(X_sparse).astype(np.float32)
    return X_reduced, svd


def build_and_save_all_features(
    n_components: int = config.LSA_COMPONENTS, models_dir: Path = config.MODELS_DIR
) -> dict:
    """Tạo và lưu đủ 3 ma trận đặc trưng: bow_lsa, tfidf_lsa, embedding.

    Luôn tạo lại từ articles_clean.csv (chỉ mất vài giây) thay vì dùng lại file cũ trong models/,
    để khi tiền xử lý hoặc config thay đổi thì đặc trưng không bị lỗi thời.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Bước 3: BoW + TF-IDF (build_and_save_all luôn lưu vào config.MODELS_DIR)
    build_bow_tfidf()

    stats = {}

    # 1. BoW + LSA
    X_bow, _ = load_matrix("bow")
    X_bow_lsa, svd_bow = fit_lsa(X_bow, n_components=n_components)
    np.save(models_dir / "bow_lsa_matrix.npy", X_bow_lsa)
    joblib.dump(svd_bow, models_dir / "bow_lsa_svd.joblib")
    stats["bow_lsa"] = {
        "shape": list(X_bow_lsa.shape),
        "explained_variance_ratio_sum": round(float(svd_bow.explained_variance_ratio_.sum()), 4),
    }

    # 2. TF-IDF + LSA
    X_tfidf, _ = load_matrix("tfidf")
    X_tfidf_lsa, svd_tfidf = fit_lsa(X_tfidf, n_components=n_components)
    np.save(models_dir / "tfidf_lsa_matrix.npy", X_tfidf_lsa)
    joblib.dump(svd_tfidf, models_dir / "tfidf_lsa_svd.joblib")
    stats["tfidf_lsa"] = {
        "shape": list(X_tfidf_lsa.shape),
        "explained_variance_ratio_sum": round(float(svd_tfidf.explained_variance_ratio_.sum()), 4),
    }

    # 3. Word Embedding (số chiều riêng, không phụ thuộc số chiều LSA)
    stats["embedding"] = build_and_save_embedding(vector_size=config.W2V_SIZE, models_dir=models_dir)

    return stats


def _expected_shape(name: str, n_components: int) -> tuple[int, int]:
    """Kích thước ma trận phải có: (số bài trong articles_clean.csv, số chiều)."""
    n_docs = len(pd.read_csv(config.ARTICLES_CLEAN_CSV, usecols=["id"]))
    if name == "embedding":
        return n_docs, config.W2V_SIZE
    return n_docs, min(n_components, n_docs - 1)  # fit_lsa giới hạn tối đa n_docs - 1


def load_features(
    name: str, n_components: int | None = None, models_dir: Path = config.MODELS_DIR
) -> np.ndarray:
    """Hàm chuẩn cho Bước 5 nạp ma trận đặc trưng dày (chưa chuẩn hóa L2).

    Args:
        name: Một trong các giá trị {'bow_lsa', 'tfidf_lsa', 'embedding'}.
        n_components: Số chiều LSA mong muốn (bỏ qua với 'embedding').
            Mặc định config.LSA_COMPONENTS.
    Returns:
        np.ndarray kích thước (N_articles, D), thứ tự dòng khớp articles_clean.csv.

    Chưa có file, hoặc file trong models/ sai kích thước (khác số chiều mong muốn, hoặc khác số bài
    hiện tại) thì tạo lại cả 3 ma trận. Nhờ vậy mọi máy nhận cùng một ma trận, không phụ thuộc
    trước đó đã chạy notebook nào.
    """
    valid_names = {"bow_lsa", "tfidf_lsa", "embedding"}
    if name not in valid_names:
        raise ValueError(f"Tên đặc trưng '{name}' không hợp lệ. Chọn một trong: {valid_names}")

    n_components = config.LSA_COMPONENTS if n_components is None else n_components
    expected = _expected_shape(name, n_components)
    file_path = models_dir / f"{name}_matrix.npy"

    if not file_path.exists():
        # models/ không đi theo git: máy mới clone về thì tự tạo đủ 3 ma trận
        print(f"Chưa có {file_path.name}, đang tạo đặc trưng {n_components} chiều...")
        build_and_save_all_features(n_components=n_components, models_dir=models_dir)
    else:
        shape = np.load(file_path, mmap_mode="r").shape
        if shape != expected:
            print(f"{file_path.name} có kích thước {shape}, cần {expected}: đang tạo lại đặc trưng...")
            build_and_save_all_features(n_components=n_components, models_dir=models_dir)

    X = np.load(file_path)
    assert X.shape == expected, f"{name}: kích thước {X.shape}, mong đợi {expected}"
    return X


if __name__ == "__main__":
    import json
    print(json.dumps(build_and_save_all_features(n_components=config.LSA_COMPONENTS), ensure_ascii=False, indent=2))