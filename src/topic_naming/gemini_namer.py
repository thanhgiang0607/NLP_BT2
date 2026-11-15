"""Sinh tên chủ đề và mô tả cho từng cụm bằng Gemini (cấu hình tốt nhất: BoW + LSA + K-Means, k = 6).

Quy trình (theo PHAN_CONG.md mục Việc của TV5):
    1. Lấy kết quả phân cụm của cấu hình tốt nhất: `bow_lsa` (10 chiều) + `kmeans`, `k = 6`.
    2. Trích top 15 từ khóa TF-IDF và 5 bài viết gần tâm cụm nhất (tiêu đề + đoạn đầu ~120 từ).
    3. Gửi prompt tới Gemini (trả về JSON `{"name": "...", "description": "..."}`).
       Không đưa danh sách 6 nhãn thật vào prompt để việc đối chiếu với nhãn thật có ý nghĩa.
    4. Lưu `outputs/topic_names/bow_lsa_kmeans_k6.json` và `outputs/tables/topic_vs_label.csv`.

Cách chạy:
    python -m src.topic_naming.gemini_namer

Phụ trách: Xuân Hoàng
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os

from dotenv import load_dotenv
from google import genai
import pandas as pd

from src import config
from src.clustering.kmeans import run_kmeans
from src.evaluation.metrics import contingency
from src.features.lsa import load_features
from src.representation.bow_tfidf import load_corpus, load_matrix
from src.topic_naming.keywords import nearest_articles, top_keywords

# Lưu ý: KHÔNG đưa danh sách 6 nhãn thật vào prompt, nếu không Gemini chỉ chép lại nhãn gốc
# làm mất ý nghĩa của bước so sánh với nhãn thật (topic_vs_label.csv).
PROMPT = (
    "Bạn là biên tập viên báo. Dưới đây là các từ khóa TF-IDF nổi bật và 5 bài viết tiêu biểu "
    "gần tâm nhất trong một cụm tin tức tiếng Việt.\n"
    "Từ khóa TF-IDF: {kw}\n\n"
    "Các bài viết tiêu biểu:\n{articles}\n\n"
    "Hãy đặt một tên chủ đề ngắn gọn (2-4 từ), phản ánh điểm chung về nội dung của toàn cụm "
    "và mô tả súc tích (1-2 câu). Ưu tiên cách gọi có tính khái quát phù hợp với phạm vi nội dung các bài viết.\n"
    'Chỉ trả về JSON hợp lệ theo định dạng: {{"name": "<tên chủ đề 2-4 từ>", "description": "<1-2 câu>"}}'
)
MODELS = ("gemini-3.5-flash", "gemini-flash-latest", "gemini-3.8-flash")


def _format_articles(articles: list[dict]) -> str:
    """Định dạng tiêu đề và đoạn đầu (~120 từ) của các bài gần tâm để đưa vào prompt."""
    return "\n".join(
        f"{i}. Tiêu đề: {a['title']}\n   Đoạn đầu: {a['snippet']}"
        for i, a in enumerate(articles, 1)
    )


def name_cluster(keywords: list[str], articles: list[dict]) -> dict[str, str]:
    """Gọi Gemini sinh tên chủ đề (2-4 từ) và mô tả (1-2 câu) cho một cụm."""
    load_dotenv(config.ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Chưa cấu hình GEMINI_API_KEY trong file .env")

    client = genai.Client(api_key=api_key)
    prompt = PROMPT.format(kw=", ".join(keywords), articles=_format_articles(articles))
    last_err = None
    for model_name in MODELS:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            data = json.loads(resp.text)
            return {
                "name": str(data["name"]).strip(),
                "description": str(data["description"]).strip(),
            }
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Không gọi được Gemini API: {last_err}") from last_err


def run_topic_naming(k: int = 6) -> tuple[dict[str, dict], object]:
    """Sinh tên chủ đề bằng Gemini cho cấu hình tốt nhất (bow_lsa + kmeans, k = 6)."""
    df, _ = load_corpus()
    X_tfidf, v_tfidf = load_matrix("tfidf")
    X = load_features("bow_lsa")
    labels, centers, _ = run_kmeans(X, k)

    kw = top_keywords(X_tfidf, v_tfidf, labels, n=15)
    near_arts = nearest_articles(X, labels, centers, df, n=5, snippet_words=120)
    clusters = sorted(kw)

    # Gọi song song cho 6 cụm để rút ngắn thời gian chờ mạng
    with ThreadPoolExecutor(max_workers=len(clusters)) as pool:
        results = list(pool.map(lambda c: name_cluster(kw[c], near_arts[c]), clusters))

    names = {
        str(c): {
            "name": res["name"],
            "description": res["description"],
            "keywords": kw[c],
        }
        for c, res in zip(clusters, results)
    }
    return names, labels


def main() -> None:
    """Lưu bow_lsa_kmeans_k6.json và xuất bảng đối chiếu topic_vs_label.csv."""
    df, _ = load_corpus()
    names, labels = run_topic_naming(k=6)

    config.TOPIC_NAMES_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.TOPIC_NAMES_DIR / "bow_lsa_kmeans_k6.json"
    json_path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")

    # Dùng contingency() trong src/evaluation/metrics.py để lấy chủ đề thật chiếm đa số mỗi cụm
    ct = contingency(df["true_label"], labels)
    rows = []
    for cid in ct.index:
        row = ct.loc[cid]
        so_bai = int(row.sum())
        rows.append({
            "cluster": int(cid),
            "gemini_name": names[str(int(cid))]["name"],
            "chu_de_chiem_da_so": str(row.idxmax()),
            "ti_le": round(float(row.max() / so_bai), 4),
        })

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.TABLES_DIR / "topic_vs_label.csv", index=False, encoding="utf-8")
    print(f"Đã lưu {json_path} và outputs/tables/topic_vs_label.csv.")


if __name__ == "__main__":
    main()
