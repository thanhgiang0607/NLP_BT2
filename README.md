# NLP_BT2 – Clustering & Topic Modeling tin tức tiếng Việt

Phân cụm bài báo tiếng Việt (≥ 6 chủ đề) với BoW / TF-IDF / Word Embedding, LSA, K-Means / K-Medoids, sinh tên chủ đề bằng Gemini và demo bằng Streamlit.

- Kế hoạch chi tiết: [KE_HOACH.md](KE_HOACH.md)
- Phân công: [PHAN_CONG.md](PHAN_CONG.md)

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # rồi điền GEMINI_API_KEY
```

> Khuyến nghị Python 3.10–3.12 (một số thư viện như `scikit-learn-extra`, `underthesea` chưa hỗ trợ tốt bản mới hơn).

## Chạy

```bash
python -m src.pipeline                 # gộp -> tiền xử lý -> đặc trưng -> 6 thí nghiệm -> kiểm tra tên chủ đề (~10 giây)
python -m src.pipeline --from cluster  # chỉ chạy lại từ bước phân cụm
python -m src.pipeline --crawl         # crawl lại dữ liệu mới trước (ghi đè data/raw, cần mạng)
python -m src.pipeline --gemini        # sinh lại tên chủ đề bằng Gemini (cần GEMINI_API_KEY trong .env)

python app/streamlit_app.py            # demo (tự dùng Python của .venv)
```

Pipeline in tổng kết cuối và cảnh báo nếu kết quả lệch với kết quả đã lưu: nhãn cụm khác `outputs/tables/cluster_labels.csv`, hoặc tên chủ đề Gemini không còn khớp cụm. Kết quả chính thức (quét số chiều LSA và k) vẫn nằm ở `notebooks/03_clustering_experiments.ipynb`.

## Cấu trúc thư mục

```
data/        raw/ (crawl theo chủ đề), processed/ (articles.csv, articles_clean.csv), stopwords/
src/         crawl/, preprocessing/, representation/, features/, clustering/,
             evaluation/, visualization/, topic_naming/, config.py, pipeline.py
notebooks/   EDA, thử nghiệm
models/      vectorizer, SVD, Word2Vec, model cụm (không commit)
outputs/     figures/, tables/, topic_names/
app/         streamlit_app.py
bao_cao_ket_qua/  báo cáo kết quả từng phần (01_thu_thap_du_lieu.md, ...)
reports/     báo cáo, slide, video demo dự phòng
tests/
```

## Quy ước làm việc

- Mỗi người làm trên branch riêng (`feat/crawl`, `feat/preprocess`, `feat/lsa`, `feat/clustering`, `feat/demo`), merge vào `main` qua Pull Request.
- Dữ liệu: CSV UTF-8, cột `id, title, content, url, source, date, true_label`. `true_label` chỉ dùng để đánh giá.
- Đường dẫn và tham số chung lấy từ `src/config.py`, seed cố định `RANDOM_STATE = 42`.
- Không commit `.env` và file model lớn.
