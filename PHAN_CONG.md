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
| Ma trận embedding + LSA (BoW + LSA, TF-IDF + LSA), chọn số chiều | 3 ma trận đặc trưng lưu trong `models/`, `notebooks/02_representation_lsa.ipynb` (biểu đồ explained variance) |

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

## Chưa phân công

- **Bước 9 – Báo cáo, slide, thuyết trình:** đề xuất mỗi người viết phần báo cáo + 3–4 slide cho mảng mình phụ trách, một người tổng hợp và chỉnh format.
