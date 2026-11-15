"""Chạy toàn bộ pipeline bằng một lệnh: dữ liệu -> tiền xử lý -> đặc trưng -> 6 thí nghiệm -> tên chủ đề.

Các bước (theo thứ tự):
    crawl     Crawl 6 chủ đề -> data/raw/*.json              (chỉ chạy khi có --crawl, vì ghi đè dữ liệu)
    merge     Gộp + lọc trùng -> data/processed/articles.csv
    clean     Tách từ, bỏ stopwords -> data/processed/articles_clean.csv
    features  BoW, TF-IDF, BoW+LSA, TF-IDF+LSA, Word2Vec -> models/
    cluster   6 thí nghiệm (3 biểu diễn x 2 mô hình) với số chiều LSA trong config, k = 6
              -> outputs/tables/pipeline_results.csv; kiểm tra cấu hình tốt nhất khớp cluster_labels.csv
    topics    Kiểm tra tên chủ đề Gemini (outputs/topic_names/) còn khớp với cụm hiện tại;
              có --gemini thì gọi Gemini sinh lại (cần GEMINI_API_KEY trong .env)

Cách chạy:
    python -m src.pipeline                      # merge -> topics, dùng dữ liệu đã crawl sẵn
    python -m src.pipeline --from features      # bỏ qua các bước trước (dữ liệu đã có)
    python -m src.pipeline --crawl              # crawl lại từ đầu (ghi đè data/raw, cần mạng)
    python -m src.pipeline --gemini             # sinh lại tên chủ đề bằng Gemini

Pipeline không ghi đè kết quả chính thức của notebook 03 (experiments.csv, cluster_labels.csv,
quét số chiều LSA và k) mà chỉ chạy lại 6 thí nghiệm ở cấu hình đã chốt và đối chiếu với chúng.

Phụ trách: Thanh Giang
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config

STEPS = ["crawl", "merge", "clean", "features", "cluster", "topics"]
REPS = ["bow_lsa", "tfidf_lsa", "embedding"]
BEST = {"representation": "bow_lsa", "model": "kmeans", "k": 6}
TOPIC_JSON = config.TOPIC_NAMES_DIR / "bow_lsa_kmeans_k6.json"


def _header(step: str) -> float:
    print(f"\n{'=' * 70}\n[{STEPS.index(step) + 1}/{len(STEPS)}] {step.upper()}\n{'=' * 70}")
    return time.time()


def step_crawl(per_topic: int) -> dict:
    from src.crawl.crawler import crawl_topic

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for topic in config.TOPICS:
        print(f"\n--- {topic} ({config.TOPICS[topic]}) ---")
        articles = crawl_topic(topic, per_topic)
        out = config.RAW_DIR / f"{topic}.json"
        out.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
        counts[topic] = len(articles)
    return {"so_bai_moi_chu_de": counts}


def step_merge() -> dict:
    from src.crawl.merge_data import merge

    df = merge()
    return {"so_bai": len(df), "so_chu_de": df["true_label"].nunique()}


def step_clean() -> dict:
    from src.preprocessing.build_clean import build_clean

    df = build_clean()
    n = df["tokens"].str.split().str.len()
    return {"so_bai": len(df), "token_tb_moi_bai": round(float(n.mean()), 1)}


def step_features() -> dict:
    from src.features.lsa import build_and_save_all_features

    stats = build_and_save_all_features(n_components=config.LSA_COMPONENTS)
    return {name: s.get("shape", [s.get("so_bai"), s.get("so_chieu")]) for name, s in stats.items()}


def step_cluster() -> dict:
    from src.clustering.kmeans import run_kmeans
    from src.clustering.kmedoids import run_kmedoids
    from src.evaluation.metrics import evaluate
    from src.features.lsa import load_features

    df = pd.read_csv(config.ARTICLES_CLEAN_CSV)
    y = df["true_label"]
    runners = {"kmeans": run_kmeans, "kmedoids": run_kmedoids}
    k = BEST["k"]

    rows, best_labels = [], None
    for rep in REPS:
        X = load_features(rep)
        for model, run in runners.items():
            labels = run(X, k)[0]
            rows.append({"representation": rep, "n_components": X.shape[1], "model": model, "k": k,
                         **evaluate(X, labels, y)})
            if (rep, model) == (BEST["representation"], BEST["model"]):
                best_labels = np.asarray(labels)

    res = pd.DataFrame(rows).round(4)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.TABLES_DIR / "pipeline_results.csv", index=False)
    print(res.to_string(index=False))

    # Đối chiếu cấu hình tốt nhất với nhãn cụm chính thức của notebook 03 (TV4)
    ref_path = config.TABLES_DIR / "cluster_labels.csv"
    match = None
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        ref = ref[(ref["representation"] == BEST["representation"]) & (ref["model"] == BEST["model"])
                  & (ref["k"] == k)].set_index("id")
        if len(ref) == len(df) and set(ref.index) == set(df["id"]):
            match = bool((ref.loc[df["id"], "cluster"].to_numpy() == best_labels).all())
    if match is False:
        print("CẢNH BÁO: nhãn cụm cấu hình tốt nhất KHÁC cluster_labels.csv "
              "(dữ liệu hoặc config đã đổi) -> chạy lại notebooks/03_clustering_experiments.ipynb.")
    elif match is None:
        print("CẢNH BÁO: không đối chiếu được với cluster_labels.csv (thiếu file hoặc khác tập bài).")

    best = res[(res["representation"] == BEST["representation"]) & (res["model"] == BEST["model"])].iloc[0]
    return {"cau_hinh_tot_nhat": f"{BEST['representation']} ({int(best['n_components'])} chiều) + "
                                  f"{BEST['model']}, k = {k}",
            "ari": float(best["ari"]), "nmi": float(best["nmi"]), "purity": float(best["purity"]),
            "khop_cluster_labels_csv": match}


def step_topics(use_gemini: bool) -> dict:
    from src.features.lsa import load_features
    from src.clustering.kmeans import run_kmeans
    from src.representation.bow_tfidf import load_matrix
    from src.topic_naming.keywords import top_keywords

    if use_gemini:
        from src.topic_naming.gemini_namer import main as gemini_main

        gemini_main()

    if not TOPIC_JSON.exists():
        print(f"CẢNH BÁO: chưa có {TOPIC_JSON.relative_to(config.ROOT)} -> chạy với --gemini. "
              "App vẫn chạy được với tên tạm ghép từ từ khóa.")
        return {"ten_chu_de": "chưa có"}

    # Tên chủ đề lưu theo số thứ tự cụm: kiểm tra từ khóa trong JSON còn khớp cụm hiện tại không
    names = json.loads(TOPIC_JSON.read_text(encoding="utf-8"))
    labels = run_kmeans(load_features(BEST["representation"]), BEST["k"])[0]
    X_tfidf, v_tfidf = load_matrix("tfidf")
    kw_now = top_keywords(X_tfidf, v_tfidf, labels, n=15)
    stale = [c for c in kw_now if names.get(str(c), {}).get("keywords") != kw_now[c]]

    for c in sorted(kw_now):
        info = names.get(str(c), {})
        print(f"  Cụm {c} ({int((labels == c).sum()):>2} bài): {info.get('name', '?')}")
    if stale:
        print(f"CẢNH BÁO: tên chủ đề của cụm {stale} không còn khớp từ khóa hiện tại "
              "(dữ liệu hoặc config đã đổi) -> chạy lại với --gemini.")
    return {"ten_chu_de": "khớp" if not stale else f"lệch ở cụm {stale}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Chạy toàn bộ pipeline phân cụm tin tức tiếng Việt")
    ap.add_argument("--from", dest="start", choices=STEPS[1:], default="merge",
                    help="bắt đầu từ bước này (mặc định: merge)")
    ap.add_argument("--crawl", action="store_true", help="crawl lại dữ liệu trước (ghi đè data/raw, cần mạng)")
    ap.add_argument("--per-topic", type=int, default=config.ARTICLES_PER_TOPIC, help="số bài mỗi chủ đề khi crawl")
    ap.add_argument("--gemini", action="store_true", help="gọi Gemini sinh lại tên chủ đề (cần GEMINI_API_KEY)")
    args = ap.parse_args()

    steps = STEPS[STEPS.index(args.start):]
    if args.crawl:
        steps = STEPS[:]
    actions = {
        "crawl": lambda: step_crawl(args.per_topic),
        "merge": step_merge,
        "clean": step_clean,
        "features": step_features,
        "cluster": step_cluster,
        "topics": lambda: step_topics(args.gemini),
    }

    summary, t_all = {}, time.time()
    for step in steps:
        t = _header(step)
        summary[step] = actions[step]()
        print(f"-> xong sau {time.time() - t:.1f}s")

    print(f"\n{'=' * 70}\nTỔNG KẾT ({time.time() - t_all:.1f}s)\n{'=' * 70}")
    for step, info in summary.items():
        print(f"  {step:<9} {info}")


if __name__ == "__main__":
    main()
