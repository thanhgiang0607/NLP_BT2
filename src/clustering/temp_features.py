"""Hàm tạm cho TV4 chạy thử trong lúc chờ TV3.

Cùng tên, cùng kiểu trả về với `src.features.lsa.load_features`. Khi TV3 xong chỉ cần đổi import:
    from src.features.lsa import load_features
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD

from src import config
from src.representation.bow_tfidf import build_bow, build_tfidf, load_corpus


def load_features(name: str) -> np.ndarray:
    """name ∈ {"bow_lsa", "tfidf_lsa"}. Trả về ma trận dày (119, 100), chưa chuẩn hóa."""
    builders = {"bow_lsa": build_bow, "tfidf_lsa": build_tfidf}
    if name not in builders:
        raise NotImplementedError(f"'{name}': chờ TV3")
    _, docs = load_corpus()
    X, _ = builders[name](docs)
    svd = TruncatedSVD(n_components=config.LSA_COMPONENTS, random_state=config.RANDOM_STATE)
    return svd.fit_transform(X)
