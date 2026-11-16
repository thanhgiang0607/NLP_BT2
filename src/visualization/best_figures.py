"""Vẽ 3 hình theo cụm của cấu hình tốt nhất (BoW + LSA + K-Means, k = 6) cho báo cáo và slide.

Đầu ra (outputs/figures/):
    scatter_tsne_best.png   t-SNE trên BoW + LSA, tô màu theo cụm
    cm_best.png             confusion matrix cụm × chủ đề thật
    wordcloud_best.png      wordcloud mỗi cụm (tần suất × idf)

Nhãn cụm lấy từ outputs/tables/cluster_labels.csv (kết quả chính thức của notebook 03), tên cụm
lấy từ outputs/topic_names/bow_lsa_kmeans_k6.json (Gemini). Mỗi cụm được ghi "Cụm i · <tên>",
và cùng một cụm có cùng màu ở cả 3 hình.

Cách chạy:
    python -m src.visualization.best_figures

Phụ trách: Thanh Hằng (hàm vẽ trong plots.py)
"""
import json

import pandas as pd

from src import config
from src.features.lsa import load_features
from src.representation.bow_tfidf import load_corpus, load_matrix
from src.visualization import plots

BEST = {"representation": "bow_lsa", "model": "kmeans", "k": 6}
TOPIC_JSON = config.TOPIC_NAMES_DIR / "bow_lsa_kmeans_k6.json"


def best_cluster_names(df: pd.DataFrame) -> pd.Series:
    """Nhãn cụm của cấu hình tốt nhất cho từng bài (theo thứ tự df), dạng "Cụm i · <tên Gemini>"."""
    lab = pd.read_csv(config.TABLES_DIR / "cluster_labels.csv")
    lab = lab[(lab["representation"] == BEST["representation"]) & (lab["model"] == BEST["model"])
              & (lab["k"] == BEST["k"])].set_index("id")["cluster"]
    if set(lab.index) != set(df["id"]):
        raise ValueError("cluster_labels.csv không khớp tập bài hiện tại -> chạy lại notebook 03.")
    clusters = lab.loc[df["id"]].to_numpy()

    names = json.loads(TOPIC_JSON.read_text(encoding="utf-8")) if TOPIC_JSON.exists() else {}
    return pd.Series([f"Cụm {c} · {names.get(str(c), {}).get('name', '')}".rstrip(" ·") for c in clusters],
                     index=df.index)


def main() -> None:
    df, docs = load_corpus()
    labels = best_cluster_names(df)
    X = load_features(BEST["representation"])  # 10 chiều theo config.LSA_COMPONENTS
    _, v_tfidf = load_matrix("tfidf")
    cfg = f"BoW + LSA {X.shape[1]} chiều + K-Means, k = {BEST['k']}"

    plots.plot_scatter_2d(X, labels, "tsne", title=f"t-SNE theo cụm ({cfg})",
                          name="scatter_tsne_best", legend_title="Cụm")
    plots.plot_confusion_matrix(df["true_label"], labels, name="cm_best",
                                title=f"Cụm so với chủ đề thật ({cfg})")
    plots.plot_wordcloud_grid(docs, labels, name="wordcloud_best", weights=plots.idf_weights(v_tfidf),
                              title=f"Wordcloud theo cụm, tên do Gemini đặt ({cfg})")
    for n in ("scatter_tsne_best", "cm_best", "wordcloud_best"):
        print("Đã lưu", (config.FIGURES_DIR / f"{n}.png").relative_to(config.ROOT))


if __name__ == "__main__":
    main()
