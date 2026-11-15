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
| Bảng so sánh 6 thí nghiệm + cấu hình tốt nhất | `outputs/tables/experiments.csv`, `outputs/tables/cluster_labels.csv`, `bao_cao_ket_qua/04_phan_cum_danh_gia.md` |

Chi tiết từng bước xem [Việc của TV4](#việc-của-tv4-phân-cụm--đánh-giá).

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
| Code K-Means, K-Medoids (khoảng cách cosine) | Không |
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

### 4. Thư viện K-Medoids: dùng `kmedoids`

Đã thử: `scikit-learn-extra` cài được nhưng **import lỗi** (`numpy.core.multiarray failed to import`) vì được biên dịch cho numpy 1.x, còn nhóm dùng numpy 2.x. Đã đổi sang thư viện **`kmedoids`** trong `requirements.txt` (chạy tốt, kết quả lặp lại được). Cách dùng xem bước 2 bên dưới.

## Việc của TV4 (phân cụm + đánh giá)

Bắt đầu được ngay: `load_features` của TV3 đã có trên `main`, máy mới clone tự tạo đủ 3 ma trận. Cài thư viện: `pip install -r requirements.txt` (có `kmedoids`).

**1. `src/evaluation/metrics.py`: chỉ số đánh giá**
- Chuyển `contingency()` và `purity()` từ `src/visualization/plots.py` sang đây; `plots.py` import lại từ file này.
- Viết `evaluate(X, labels, true_labels) -> dict` trả về 6 chỉ số:
  - Nội tại (không cần nhãn): Silhouette (`metric="cosine"`), Davies-Bouldin, Calinski-Harabasz.
  - So với nhãn gốc: ARI, NMI, Purity.
- DB và CH đo bằng khoảng cách Euclid: tính trên ma trận **đã chuẩn hóa L2** để nhất quán với cosine.

**2. `src/clustering/kmeans.py`, `src/clustering/kmedoids.py`**
- `run_kmeans(X, k)`: chuẩn hóa L2 → `KMeans(k, n_init=10, random_state=config.RANDOM_STATE)`.
- `run_kmedoids(X, k)`: khoảng cách cosine → K-Medoids PAM:

  ```python
  import kmedoids
  from sklearn.metrics.pairwise import cosine_distances

  D = cosine_distances(X)  # (119, 119)
  km = kmedoids.KMedoids(k, metric="precomputed", method="pam",
                         random_state=config.RANDOM_STATE).fit(D)
  km.labels_, km.medoid_indices_
  ```
- Cả hai trả về nhãn cụm **và** tâm cụm (K-Means) / chỉ số bài làm tâm (K-Medoids). TV5 cần phần này để chọn bài gần tâm gửi Gemini.

**3. `src/clustering/select_k.py`: chọn k**
- Chạy k ∈ `config.K_RANGE` (2–12), vẽ Elbow (inertia) và Silhouette theo k → `outputs/figures/`.

**4. `notebooks/03_clustering_experiments.ipynb`: thí nghiệm**
- 3 biểu diễn × 2 mô hình = 6 thí nghiệm, mỗi thí nghiệm chạy với **k chọn ở bước 3** và **k = 6** (số chủ đề thật).
- **Quét số chiều LSA** {10, 20, 30, 50, 80} cho BoW+LSA và TF-IDF+LSA (`build_and_save_all_features(n_components=...)`). Đo nhanh: TF-IDF+LSA 20 chiều tốt hơn hẳn 80 chiều (ARI 0,443 so với 0,335), xem báo cáo phần 3.
- Lưu kết quả:

  | File | Nội dung |
  |---|---|
  | `outputs/tables/experiments.csv` | Mỗi dòng 1 thí nghiệm: `representation, n_components, model, k` + 6 chỉ số |
  | `outputs/tables/cluster_labels.csv` | `id, representation, model, k, cluster` (TV2, TV5 dùng) |

**5. Chọn cấu hình tốt nhất + báo cáo `bao_cao_ket_qua/04_phan_cum_danh_gia.md`**
- Không chọn chỉ theo ARI/NMI: nhãn gốc có nhiễu (bài nổ gas xếp vào Sức khỏe, Nvidia rải ở 3 chủ đề, xem báo cáo phần 1). Xét cùng lúc chỉ số nội tại, chỉ số so với nhãn và mức dễ hiểu của cụm.
- Mức tham khảo từ phép thử nhanh: K-Means k = 6 cho ARI khoảng 0,33–0,47; K-Medoids trên embedding ARI 0,39. Đây là mức hợp lý với 119 bài có nhãn nhiễu, không phải dấu hiệu code sai.

**Bàn giao**

| Đầu ra của TV4 | Ai dùng |
|---|---|
| Nhãn cụm của cấu hình tốt nhất | TV2: scatter, wordcloud, confusion matrix theo cụm |
| Nhãn cụm + tâm cụm / bài làm tâm | TV5: top từ khóa, 3–5 bài gần tâm gửi Gemini |
| `experiments.csv` | Báo cáo, slide |

## Chưa phân công

- **Bước 9 – Báo cáo, slide, thuyết trình:** đề xuất mỗi người viết phần báo cáo + 3–4 slide cho mảng mình phụ trách, một người tổng hợp và chỉnh format.
