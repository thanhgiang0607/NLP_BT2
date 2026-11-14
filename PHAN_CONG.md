# Phân công công việc

> Điền tên thành viên vào cột **Người phụ trách**. Chi tiết từng bước xem [KE_HOACH.md](KE_HOACH.md).

## Tổng quan

| # | Mảng công việc | Bước liên quan | Người phụ trách |
|---|---|---|---|
| 1 | Code: tạo repo + tìm data | Bước 0, 1 (+ tích hợp, kiểm thử) | _TV1_ |
| 2 | Code: tiền xử lý, BoW/TF-IDF, trực quan | Bước 2, 3 (BoW/TF-IDF), 6 (hình) | _TV2_ |
| 3 | Code: Word Embedding + LSA/LDA | Bước 3 (Embedding), 4 | _TV3_ |
| 4 | Code: phân cụm + đánh giá | Bước 5, 6 (chỉ số, bảng) | _TV4_ |
| 5 | Code: sinh chủ đề + demo | Bước 7, 8 | _TV5_ |

---

## 1. Code: tạo repo + tìm data — _TV1_

| Công việc | Đầu ra (file) |
|---|---|
| Tạo repo, cấu trúc thư mục | Repo GitHub, `README.md`, `requirements.txt`, `.gitignore`, `.env.example`, `src/config.py` |
| Script crawl data, crawl đủ 6 chủ đề | `src/crawl/crawler.py`, `src/crawl/sources.py`, `data/raw/<chu_de>.json` |
| Gộp file dữ liệu | `src/crawl/merge_data.py` → `data/processed/articles.csv` (đã lọc trùng, bài < 150 từ, bài chỉ có video/ảnh) |
| Review code + ghép pipeline sau khi xong | `src/pipeline.py`, review các Pull Request |
| Kiểm thử demo | Chạy thử app trên máy sạch, báo lỗi cho TV5 |

## 2. Code: tiền xử lý, BoW/TF-IDF, trực quan — _TV2_

| Công việc | Đầu ra (file) |
|---|---|
| Code tiền xử lý (tách từ, stopwords) | `src/preprocessing/*.py`, `data/stopwords/vietnamese_stopwords.txt` → `data/processed/articles_clean.csv` |
| EDA | `notebooks/01_eda.ipynb` (số bài/chủ đề, độ dài bài, top từ) |
| Ma trận BoW, TF-IDF | `src/representation/bow_tfidf.py`, vectorizer lưu trong `models/` |
| Trực quan: t-SNE/UMAP, wordcloud, confusion matrix | `src/visualization/plots.py` → `outputs/figures/` |

## 3. Code: Word Embedding + LSA/LDA — _TV3_

| Công việc | Đầu ra (file) |
|---|---|
| Code Word2Vec/PhoW2V + LSA | `src/representation/embedding.py`, `src/features/lsa.py` (+ `src/features/lda.py` nếu làm thêm) |
| Ma trận embedding + LSA (BoW + LSA, TF-IDF + LSA), chọn số chiều | Hàm `load_features(name)` trong `src/features/lsa.py` (tạo + lưu 3 ma trận vào `models/`), `notebooks/02_representation_lsa.ipynb` (biểu đồ explained variance). Xem [Quy ước trao đổi TV3 ↔ TV4](#quy-ước-trao-đổi-tv3--tv4) |

## 4. Code: phân cụm + đánh giá — _TV4_

| Công việc | Đầu ra (file) |
|---|---|
| Code K-Means, K-Medoids, các chỉ số đánh giá | `src/clustering/*.py`, `src/evaluation/metrics.py` |
| Chạy đủ 6 thí nghiệm, chọn k | `notebooks/03_clustering_experiments.ipynb`, biểu đồ Elbow/Silhouette trong `outputs/figures/` |
| Bảng so sánh 6 thí nghiệm + cấu hình tốt nhất | `outputs/tables/experiments.csv`, nhãn cụm của cấu hình tốt nhất |

## 5. Code: sinh chủ đề + demo — _TV5_

| Công việc | Đầu ra (file) |
|---|---|
| Prompt Gemini + khung app Streamlit | `src/topic_naming/gemini_namer.py`, `src/topic_naming/keywords.py`, `src/topic_naming/spacy_np.py`, khung `app/streamlit_app.py` |
| Sinh tên chủ đề cho các cụm thật + bảng so với nhãn thật | `outputs/topic_names/*.json`, `outputs/tables/topic_vs_label.csv` |
| App Streamlit hoàn chỉnh | `app/streamlit_app.py` (chọn biểu diễn/mô hình/k, scatter, wordcloud, tên chủ đề, dự đoán bài mới/URL) |
| Thao tác máy khi demo | Demo trực tiếp + `reports/demo_backup.mp4` |

---

## Phụ thuộc giữa các mảng

```
TV1 (articles.csv)
  └─→ TV2 (articles_clean.csv, BoW/TF-IDF)
        └─→ TV3 (Embedding, BoW+LSA, TF-IDF+LSA)
              └─→ TV4 (6 thí nghiệm, nhãn cụm, bảng so sánh)
                    ├─→ TV2 (vẽ scatter, wordcloud, confusion matrix theo nhãn cụm)
                    └─→ TV5 (tên chủ đề Gemini, app hoàn chỉnh)
                          └─→ TV1 (ghép pipeline, kiểm thử demo)
```

**Để không bị chờ nhau:**

- TV1 đẩy sớm một file mẫu ~20 bài (đúng định dạng cột) để TV2 viết code tiền xử lý trước khi crawl xong.
- TV2–TV5 viết code dựa trên dữ liệu mẫu / ma trận ngẫu nhiên đúng shape, khi có dữ liệu thật chỉ cần chạy lại.
- TV5 dựng khung Streamlit và test prompt Gemini với cụm giả ngay từ đầu.

## Quy ước trao đổi TV3 ↔ TV4

TV3 và TV4 **làm song song được**. TV2 đã xong `articles_clean.csv` và BoW/TF-IDF trên `main`, nên TV3 bắt đầu ngay. Chỉ bước chạy chính thức 6 thí nghiệm của TV4 mới cần chờ TV3.

| Việc của TV4 | Cần chờ TV3? |
|---|---|
| Code K-Means, K-Medoids (`metric="cosine"`) | Không |
| Code các chỉ số: Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI, Purity | Không |
| Code chọn k (Elbow, Silhouette) | Không |
| Chạy thử toàn bộ code | Không. Tự tạo tạm TF-IDF + `TruncatedSVD(100)` từ `src/representation/bow_tfidf.py` |
| Chạy chính thức 6 thí nghiệm, lập bảng so sánh | **Có**, cần đủ 3 ma trận của TV3 |

### 1. Lấy ma trận đặc trưng qua hàm, không gửi file

Thư mục `models/` bị `.gitignore` chặn, nên file `.npy` **không đi theo git**. TV3 cung cấp hàm, TV4 gọi hàm:

```python
# src/features/lsa.py (TV3)
def load_features(name: str) -> np.ndarray:
    """name ∈ {"bow_lsa", "tfidf_lsa", "embedding"}.
    Trả về ma trận (119, d); chưa có trong models/ thì tự tạo rồi lưu."""
```

Trong lúc TV3 chưa xong, TV4 viết một hàm tạm **cùng tên, cùng kiểu trả về**, có TV3 thì chỉ đổi import.

### 2. Quy ước ma trận

| Mục | Quy ước |
|---|---|
| Thứ tự dòng | Trùng `data/processed/articles_clean.csv`: dòng i là bài `id` thứ i |
| Kiểu dữ liệu | `numpy.ndarray` dày (không dùng ma trận thưa) |
| Chuẩn hóa L2 | TV3 trả về **chưa chuẩn hóa**; TV4 tự chuẩn hóa trước K-Means. K-Medoids dùng cosine nên không cần |
| Seed | `config.RANDOM_STATE` cho SVD, Word2Vec, K-Means, K-Medoids |
| Nhãn thật | `true_label` chỉ dùng để tính ARI/NMI/Purity, **không** đưa vào ma trận |

### 3. Hàm Purity

`purity()` và `contingency()` hiện nằm trong `src/visualization/plots.py`. TV4 chuyển sang `src/evaluation/metrics.py`, rồi `plots.py` import lại từ đó, để cả nhóm dùng chung một cách tính.

### 4. Cài K-Medoids sớm

`sklearn_extra` (KMedoids) hay lỗi khi cài trên Python mới. TV4 cài thử ngay ngày đầu trong `.venv` Python 3.12 của nhóm. Không cài được thì chuyển sang thư viện `kmedoids` hoặc tự viết PAM, đừng để đến lúc chạy thí nghiệm mới phát hiện.

## Chưa phân công

- **Bước 9 – Báo cáo, slide, thuyết trình:** đề xuất mỗi người viết phần báo cáo + 3–4 slide cho mảng mình phụ trách, một người tổng hợp và chỉnh format.
