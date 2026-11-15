"""Trích xuất đặc trưng LSA (TruncatedSVD) cho BoW và TF-IDF, đồng thời cung cấp hàm load_features().

Phụ trách: Gia Khang
TODO: đang duyệt.
"""
from pathlib import Path
import joblib
import numpy as np
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
    """Tạo và lưu đủ 3 ma trận đặc trưng: bow_lsa, tfidf_lsa, embedding."""
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Tự động chạy Bước 3 nếu chưa có sẵn file bow_matrix.npz hoặc tfidf_matrix.npz
    if not (models_dir / "bow_matrix.npz").exists() or not (models_dir / "tfidf_matrix.npz").exists():
        print("Chưa tìm thấy ma trận BoW/TF-IDF, đang tự động tạo từ Bước 3...")
        build_bow_tfidf()

    stats = {}

    # 1. BoW + LSA
    X_bow, _ = load_matrix("bow", models_dir)
    X_bow_lsa, svd_bow = fit_lsa(X_bow, n_components=n_components)
    np.save(models_dir / "bow_lsa_matrix.npy", X_bow_lsa)
    joblib.dump(svd_bow, models_dir / "bow_lsa_svd.joblib")
    stats["bow_lsa"] = {
        "shape": list(X_bow_lsa.shape),
        "explained_variance_ratio_sum": round(float(svd_bow.explained_variance_ratio_.sum()), 4),
    }

    # 2. TF-IDF + LSA
    X_tfidf, _ = load_matrix("tfidf", models_dir)
    X_tfidf_lsa, svd_tfidf = fit_lsa(X_tfidf, n_components=n_components)
    np.save(models_dir / "tfidf_lsa_matrix.npy", X_tfidf_lsa)
    joblib.dump(svd_tfidf, models_dir / "tfidf_lsa_svd.joblib")
    stats["tfidf_lsa"] = {
        "shape": list(X_tfidf_lsa.shape),
        "explained_variance_ratio_sum": round(float(svd_tfidf.explained_variance_ratio_.sum()), 4),
    }

    # 3. Word Embedding (nếu chưa có file .npy thì chạy build)
    emb_path = models_dir / "embedding_matrix.npy"
    if not emb_path.exists():
        stats["embedding"] = build_and_save_embedding(vector_size=n_components, models_dir=models_dir)
    else:
        X_emb = np.load(emb_path)
        stats["embedding"] = {"shape": list(X_emb.shape)}

    return stats


def load_features(name: str, models_dir: Path = config.MODELS_DIR) -> np.ndarray:
    """Hàm chuẩn cho Bước 5 nạp ma trận đặc trưng dày (chưa chuẩn hóa L2).

    Args:
        name: Một trong các giá trị {'bow_lsa', 'tfidf_lsa', 'embedding'}.
    Returns:
        np.ndarray kích thước (N_articles, D), thứ tự dòng khớp articles_clean.csv.
    """
    valid_names = {"bow_lsa", "tfidf_lsa", "embedding"}
    if name not in valid_names:
        raise ValueError(f"Tên đặc trưng '{name}' không hợp lệ. Chọn một trong: {valid_names}")

    file_path = models_dir / f"{name}_matrix.npy"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {file_path}. Hãy chạy `python -m src.features.lsa` trước."
        )
    return np.load(file_path)


if __name__ == "__main__":
    import json
    print(json.dumps(build_and_save_all_features(n_components=config.LSA_COMPONENTS), ensure_ascii=False, indent=2))