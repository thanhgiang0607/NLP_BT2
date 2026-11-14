"""Mô hình LDA (gensim) chạy trên tập từ vựng để đối chiếu với LSA và trích xuất top từ khóa.

Phụ trách: Gia Khang
TODO: đang duyệt.
"""
from pathlib import Path
import joblib
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from src import config
from src.representation.bow_tfidf import load_corpus
from src.representation.tokens import split_tokens


def train_lda(
    num_topics: int = 6,
    passes: int = 15,
    random_state: int = config.RANDOM_STATE,
    models_dir: Path = config.MODELS_DIR,
) -> tuple[LdaModel, Dictionary, np.ndarray]:
    """Huấn luyện LDA và trả về (model, dictionary, ma trận phân phối chủ đề của văn bản)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    _, docs = load_corpus()
    tokenized_docs = [split_tokens(d) for d in docs]

    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=config.MIN_DF, no_above=config.MAX_DF)
    corpus = [dictionary.doc2bow(toks) for toks in tokenized_docs]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
    )

    # Ma trận dày (N_articles, num_topics)
    topic_matrix = np.zeros((len(corpus), num_topics), dtype=np.float32)
    for i, bow in enumerate(corpus):
        for topic_id, prob in lda.get_document_topics(bow, minimum_probability=0.0):
            topic_matrix[i, topic_id] = prob

    lda.save(str(models_dir / "lda.model"))
    joblib.dump(dictionary, models_dir / "lda_dict.joblib")
    np.save(models_dir / "lda_matrix.npy", topic_matrix)

    return lda, dictionary, topic_matrix


def get_lda_top_words(lda: LdaModel, topn: int = 10) -> dict[int, list[str]]:
    """Lấy danh sách top từ khóa đại diện cho từng topic của LDA."""
    topics = {}
    for topic_id in range(lda.num_topics):
        words = [word for word, _ in lda.show_topic(topic_id, topn=topn)]
        topics[topic_id] = words
    return topics