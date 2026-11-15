"""Word2Vec tự train trên tập tin tức; vector bài = trung bình vector từ.

Phụ trách: Gia Khang
TODO: đang duyệt.
"""
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec

from src import config
from src.representation.bow_tfidf import load_corpus
from src.representation.tokens import split_tokens


def train_word2vec(
    tokenized_docs: list[list[str]],
    vector_size: int = config.LSA_COMPONENTS,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 30,
    seed: int = config.RANDOM_STATE,
) -> Word2Vec:
    """Huấn luyện mô hình Word2Vec (Skip-gram) cố định seed."""
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # Skip-gram hiệu quả hơn trên tập dữ liệu nhỏ
        workers=1,  # Bắt buộc workers=1 để tái lập kết quả với seed
        seed=seed,
        epochs=epochs,
    )
    return model


def doc_to_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    """Tính vector trung bình cho 1 văn bản từ các token có trong từ điển."""
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def build_embedding_matrix(
    docs: list[str], model: Word2Vec
) -> np.ndarray:
    """Chuyển toàn bộ danh sách văn bản thành ma trận dày (N, vector_size)."""
    tokenized_docs = [split_tokens(doc) for doc in docs]
    matrix = np.vstack([doc_to_vector(toks, model) for toks in tokenized_docs])
    return matrix


def build_and_save_embedding(
    vector_size: int = config.LSA_COMPONENTS, models_dir: Path = config.MODELS_DIR
) -> dict:
    """Pipeline: đọc corpus -> train Word2Vec -> tạo ma trận -> lưu vào models/."""
    models_dir.mkdir(parents=True, exist_ok=True)
    _, docs = load_corpus()
    tokenized_docs = [split_tokens(d) for d in docs]

    w2v_model = train_word2vec(tokenized_docs, vector_size=vector_size)
    X_emb = build_embedding_matrix(docs, w2v_model)

    w2v_model.save(str(models_dir / "word2vec.model"))
    np.save(models_dir / "embedding_matrix.npy", X_emb)

    return {
        "so_bai": int(X_emb.shape[0]),
        "so_chieu": int(X_emb.shape[1]),
        "vocab_size": len(w2v_model.wv),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(build_and_save_embedding(), ensure_ascii=False, indent=2))