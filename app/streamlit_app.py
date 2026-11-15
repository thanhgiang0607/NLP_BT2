"""Demo Streamlit: khám phá cụm (biểu diễn, mô hình, k) và dự đoán bài báo mới.

Giao diện gồm 2 tab đúng theo KE_HOACH.md (Bước 8) và PHAN_CONG.md (Mục 5 của TV5):
    - Tab 1 (Khám phá cụm): chọn cách biểu diễn, mô hình và số cụm k; hiển thị bảng tên chủ đề,
      biểu đồ scatter 2D (t-SNE) và wordcloud từ khóa cho từng cụm.
      Với k = 6, tên chủ đề được đọc sẵn từ `outputs/topic_names/*.json` để chạy offline ổn định.
    - Tab 2 (Dự đoán bài mới): dán văn bản bài báo mới hoặc nhập URL, dự đoán bài thuộc cụm nào
      và chủ đề gì theo cấu hình tốt nhất đã chốt (BoW + LSA + K-Means, k = 6).

Cách chạy:
    streamlit run app/streamlit_app.py

Phụ trách: Xuân Hoàng
"""
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Nếu bấm Run hoặc chạy trực tiếp `python app/streamlit_app.py`, tự gọi `streamlit run` bằng Python của `.venv`
if __name__ == "__main__" and "streamlit" not in sys.modules:
    # Windows: .venv/Scripts/python.exe; macOS / Linux: .venv/bin/python
    candidates = [ROOT / ".venv/Scripts/python.exe", ROOT / ".venv/bin/python"]
    venv_py = next((p for p in candidates if p.exists()), None)
    py = str(venv_py or Path(sys.executable))
    raise SystemExit(subprocess.run([py, "-m", "streamlit", "run", __file__, *sys.argv[1:]], cwd=ROOT).returncode)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import normalize

from src import config
from src.clustering.kmeans import run_kmeans
from src.clustering.kmedoids import run_kmedoids
from src.crawl.crawler import extract_body, fetch, norm
from src.crawl.sources import SELECTORS
from src.features.lsa import load_features
from src.preprocessing.build_clean import preprocess_text
from src.preprocessing.stopwords import load_stopwords
from src.representation import bow_tfidf
from src.topic_naming.keywords import top_keywords
from src.visualization import plots


REP_LABELS = {"BoW + LSA": "bow_lsa", "TF-IDF + LSA": "tfidf_lsa", "Word Embedding": "embedding"}
MODEL_LABELS = {"K-Means": "kmeans", "K-Medoids": "kmedoids"}
DOMAIN_TO_SOURCE = {
    "vnexpress.net": "vnexpress",
    "tuoitre.vn": "tuoitre",
    "thanhnien.vn": "thanhnien",
    "dantri.com.vn": "dantri",
}


# Dùng @st.cache_resource cho model/tài nguyên và @st.cache_data cho kết quả phân cụm
# để không phải nạp lại hoặc fit lại mỗi lần người dùng đổi thao tác giao diện.
@st.cache_resource
def load_resources():
    """Nạp tập bài báo, stopwords, ma trận đặc trưng và bộ biến đổi BoW + LSA từ `models/`."""
    df, docs = bow_tfidf.load_corpus()
    stopwords = load_stopwords()
    X_reps = {rep: load_features(rep) for rep in REP_LABELS.values()}
    X_tfidf, v_tfidf = bow_tfidf.load_matrix("tfidf")
    bow_vec = joblib.load(config.MODELS_DIR / "bow_vectorizer.joblib")
    bow_svd = joblib.load(config.MODELS_DIR / "bow_lsa_svd.joblib")
    return df, docs, stopwords, X_reps, X_tfidf, v_tfidf, bow_vec, bow_svd


@st.cache_data
def get_clustering(rep: str, model_key: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Chạy K-Means hoặc K-Medoids trên ma trận đặc trưng và cache kết quả."""
    _, _, _, X_reps, _, _, _, _ = load_resources()
    runner = run_kmeans if model_key == "kmeans" else run_kmedoids
    labels, centers_or_medoids, _ = runner(X_reps[rep], k)
    return np.asarray(labels, dtype=int), np.asarray(centers_or_medoids)


def get_topic_names(rep: str, model_key: str, k: int, X_tfidf, v_tfidf, labels) -> dict[str, dict]:
    """Đọc tên chủ đề do Gemini sinh từ JSON (nếu có), hoặc ghép ngắn từ top 3 từ khóa TF-IDF."""
    path = config.TOPIC_NAMES_DIR / f"{rep}_{model_key}_k{k}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    kw = top_keywords(X_tfidf, v_tfidf, labels, n=12)
    return {
        str(c): {
            "name": " / ".join(w.replace("_", " ") for w in kw[c][:3]).title(),
            "description": f"Cụm {c} gồm các từ khóa: {', '.join(w.replace('_', ' ') for w in kw[c][:6])}.",
            "keywords": kw[c],
        }
        for c in sorted(kw)
    }


def fetch_article_from_url(url: str) -> str:
    """Tải trang bằng `crawler.fetch(url)` và lấy tiêu đề/nội dung theo `SELECTORS` (hoặc fallback `<h1>`/`<article>`)."""
    soup = fetch(url.strip())
    if soup is None:
        return ""

    host = urlparse(url).netloc.lower()
    source = next((s for d, s in DOMAIN_TO_SOURCE.items() if host == d or host.endswith(f".{d}")), None)
    title, sapo, body = "", "", ""
    if source in SELECTORS:
        sel = SELECTORS[source]
        t_el, s_el, b_el = soup.select_one(sel["title"]), soup.select_one(sel["sapo"]), soup.select_one(sel["body"])
        title = norm(t_el.get_text(" ")) if t_el else ""
        sapo = norm(s_el.get_text(" ")) if s_el else ""
        # VnExpress đặt sapo nằm bên trong thẻ thân bài -> tách ra trước để tránh lặp đoạn mở đầu
        if s_el is not None and b_el is not None and b_el in s_el.parents:
            s_el.extract()
        body = extract_body(b_el) if b_el is not None else ""

    if not title and soup.find("h1"):
        title = norm(soup.find("h1").get_text(" "))
    if not body:
        body = extract_body(soup.find("article") or soup)
    return "\n".join(part for part in (title, sapo, body) if part)


def predict_article(text: str) -> dict | None:
    """Dự đoán cụm và chủ đề cho bài báo mới theo cấu hình tốt nhất (BoW + LSA + K-Means, k = 6)."""
    _, _, stopwords, _, X_tfidf, v_tfidf, bow_vec, bow_svd = load_resources()

    # Bài mới đi qua đúng các bước như lúc huấn luyện (theo PHAN_CONG.md mục 5):
    # preprocess_text -> bow_vectorizer -> bow_lsa_svd -> normalize L2 -> cụm có tâm gần nhất
    tokens = preprocess_text(text, stopwords)
    if not tokens:
        return None

    z = normalize(bow_svd.transform(bow_vec.transform([" ".join(tokens)])))
    labels, centers = get_clustering("bow_lsa", "kmeans", 6)
    cid = int(np.argmax(z @ normalize(centers).T))

    names_map = get_topic_names("bow_lsa", "kmeans", 6, X_tfidf, v_tfidf, labels)
    return {"cluster": cid, "info": names_map.get(str(cid), {})}


def render_explore_tab(df: pd.DataFrame, docs: list[str], X_reps: dict, X_tfidf, v_tfidf) -> None:
    """Tab 1: chọn biểu diễn, mô hình, k -> hiện bảng tên chủ đề, scatter t-SNE và wordcloud mỗi cụm."""
    c1, c2, c3 = st.columns(3)
    rep_label = c1.selectbox("Cách biểu diễn", list(REP_LABELS), index=0)
    model_label = c2.selectbox("Mô hình phân cụm", list(MODEL_LABELS), index=0)
    k = c3.slider("Số cụm (k)", min_value=2, max_value=12, value=6)

    rep, model_key = REP_LABELS[rep_label], MODEL_LABELS[model_label]
    X_mat = X_reps[rep]
    labels, _ = get_clustering(rep, model_key, k)
    names_map = get_topic_names(rep, model_key, k, X_tfidf, v_tfidf, labels)

    topic_rows = []
    for cid in sorted(set(labels)):
        info = names_map.get(str(cid), {})
        topic_rows.append({
            "Cụm": int(cid),
            "Số bài": int((labels == cid).sum()),
            "Tên chủ đề": info.get("name", f"Cụm {cid}"),
            "Mô tả": info.get("description", ""),
            "Từ khóa": ", ".join(w.replace("_", " ") for w in info.get("keywords", [])[:10]),
        })

    st.subheader("Tên chủ đề của từng cụm")
    st.dataframe(pd.DataFrame(topic_rows), width="stretch", hide_index=True)

    st.subheader("Biểu đồ phân tán 2D (t-SNE)")
    fig_scatter = plots.plot_scatter_2d(
        X_mat, labels, method="tsne",
        title=f"t-SNE ({rep_label} + {model_label}, k={k})", legend_title="Cụm",
    )
    st.columns([1, 2.4, 1])[1].pyplot(fig_scatter, width="stretch")
    plt.close(fig_scatter)

    st.subheader("Wordcloud từ khóa của từng cụm")
    fig_wc = plots.plot_wordcloud_grid(
        docs, labels, weights=plots.idf_weights(v_tfidf),
        title=f"Wordcloud ({rep_label} + {model_label}, k={k})",
    )
    st.columns([1, 2.4, 1])[1].pyplot(fig_wc, width="stretch")
    plt.close(fig_wc)


def render_predict_tab() -> None:
    """Tab 2: dán bài báo mới hoặc nhập URL -> dự đoán bài thuộc cụm nào và chủ đề gì."""
    st.caption("Sử dụng cấu hình tốt nhất đã chốt: **BoW + LSA (10 chiều) + K-Means, k = 6**.")
    mode = st.radio("Nguồn bài báo", ["Dán nội dung văn bản", "Nhập URL bài báo"], horizontal=True)

    text = ""
    if mode == "Dán nội dung văn bản":
        text = st.text_area(
            "Nội dung bài báo",
            height=180,
            placeholder="Dán tiêu đề và nội dung bài báo tiếng Việt vào đây...",
        )
    else:
        url = st.text_input(
            "URL bài báo (VnExpress, Tuổi Trẻ, Thanh Niên, Dân trí...)",
            placeholder="https://vnexpress.net/...",
        )
        if url.strip():
            text = fetch_article_from_url(url)
            if text:
                st.caption(f"Đã trích {len(text.split())} từ từ URL.")
            else:
                st.warning("Không trích được nội dung từ URL này. Kiểm tra lại đường dẫn hoặc mạng.")

    if not st.button("Dự đoán chủ đề", type="primary"):
        return

    res = predict_article(text)
    if not res:
        st.warning("Không có nội dung hợp lệ để dự đoán. Vui lòng kiểm tra văn bản hoặc URL.")
        return

    info = res["info"]
    st.success(f"**Cụm dự đoán:** Cụm {res['cluster']} – **{info.get('name', 'Chưa đặt tên')}**")
    st.write("**Mô tả:**", info.get("description", ""))
    st.write("**Từ khóa của cụm:**", ", ".join(w.replace("_", " ") for w in info.get("keywords", [])[:10]))


def main() -> None:
    """Khởi chạy giao diện Streamlit gồm 2 tab: Khám phá cụm và Dự đoán bài mới."""
    st.set_page_config(page_title="Phân cụm Tin tức Tiếng Việt", layout="wide")
    st.title("Phân cụm & Sinh tên Chủ đề Tin tức Tiếng Việt")

    df, docs, _, X_reps, X_tfidf, v_tfidf, _, _ = load_resources()
    tab1, tab2 = st.tabs(["Khám phá cụm", "Dự đoán bài mới"])
    with tab1:
        render_explore_tab(df, docs, X_reps, X_tfidf, v_tfidf)
    with tab2:
        render_predict_tab()


if __name__ == "__main__":
    main()
