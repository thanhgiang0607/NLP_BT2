# Kế hoạch bài tập Clustering & Topic Modeling

## Yêu cầu đề bài

- Thu thập dữ liệu tiếng Việt (tối thiểu **50 bài viết**) về chủ đề **Tin tức**.
- Đảm bảo đa dạng: tối thiểu **5 chủ đề**, mỗi chủ đề tối thiểu **10 bài viết**.
- Tách đặc trưng bằng 1 trong 2 phương pháp: **LSA / LDA**.
- Kết hợp các phương pháp biểu diễn: **Bag-of-Words, TF-IDF, Word Embedding**.
- Sử dụng 2 mô hình phân cụm: **K-Means, K-Medoids**.
- Sử dụng công cụ sinh chủ đề cho các cụm (**spaCy / kết hợp LLM - Gemini**…).
- Có **demo**.

---

## Cấu trúc thư mục

```
NLP_BT2/
├── README.md                     # Hướng dẫn cài đặt, chạy pipeline, chạy demo
├── KE_HOACH.md                   # File này
├── PHAN_CONG.md                  # Phân công công việc
├── requirements.txt              # Thư viện dùng chung (pin version)
├── .env.example                  # GEMINI_API_KEY=...  (file .env thật KHÔNG commit)
├── .gitignore                    # .env, __pycache__, models/*.pkl lớn, data/raw nếu nặng
│
├── data/
│   ├── raw/                      # Dữ liệu crawl gốc, mỗi chủ đề 1 file
│   │   ├── thoi_su.json
│   │   ├── kinh_te.json
│   │   ├── the_thao.json
│   │   ├── giai_tri.json
│   │   ├── cong_nghe.json
│   │   └── suc_khoe.json
│   ├── processed/
│   │   ├── articles.csv          # Đã gộp + lọc trùng/ngắn: id,title,content,url,source,date,true_label
│   │   └── articles_clean.csv    # + cột tokens (đã tách từ, bỏ stopwords)
│   └── stopwords/
│       └── vietnamese_stopwords.txt   # Bộ có sẵn + từ đặc thù báo chí
│
├── src/
│   ├── config.py                 # Đường dẫn, seed, danh sách chủ đề, tham số chung
│   ├── crawl/
│   │   ├── crawler.py            # requests + BeautifulSoup / newspaper3k
│   │   ├── sources.py            # URL chuyên mục của VnExpress, Tuổi Trẻ, Thanh Niên, Dân trí
│   │   └── merge_data.py         # Gộp data/raw/* → data/processed/articles.csv, lọc
│   ├── preprocessing/
│   │   ├── clean.py              # Bỏ HTML, URL, số, ký tự đặc biệt, lowercase
│   │   ├── tokenizer.py          # underthesea / pyvi (bóng đá → bóng_đá)
│   │   └── stopwords.py          # Load + lọc stopwords
│   ├── representation/
│   │   ├── bow_tfidf.py          # CountVectorizer, TfidfVectorizer (ngram 1–2, min_df, max_df)
│   │   └── embedding.py          # Word2Vec tự train / PhoW2V / fastText → vector trung bình
│   ├── features/
│   │   ├── lsa.py                # TruncatedSVD 50–100 chiều, chọn số chiều
│   │   └── lda.py                # gensim LDA (so sánh + top từ khóa)
│   ├── clustering/
│   │   ├── kmeans.py
│   │   ├── kmedoids.py           # sklearn_extra KMedoids, metric="cosine"
│   │   └── select_k.py           # Elbow + Silhouette
│   ├── evaluation/
│   │   └── metrics.py            # Silhouette, DB, CH, ARI, NMI, Purity
│   ├── visualization/
│   │   └── plots.py              # t-SNE/UMAP scatter, wordcloud, confusion matrix
│   ├── topic_naming/
│   │   ├── keywords.py           # Top 10–15 từ mỗi cụm
│   │   ├── spacy_np.py           # spacy.blank("vi") + pyvi, đếm cụm danh từ
│   │   └── gemini_namer.py       # Prompt Gemini → tên chủ đề + mô tả
│   └── pipeline.py               # Chạy toàn bộ: data → features → 6 thí nghiệm → kết quả
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_representation_lsa.ipynb
│   ├── 03_clustering_experiments.ipynb
│   └── 04_topic_naming.ipynb
│
├── models/                       # Vectorizer, SVD, Word2Vec, model cụm đã fit (.pkl/.model)
│
├── outputs/
│   ├── figures/                  # scatter, wordcloud, confusion matrix, elbow/silhouette
│   ├── tables/                   # experiments.csv (bảng 6 thí nghiệm), topic_vs_label.csv
│   └── topic_names/              # JSON tên chủ đề Gemini sinh cho từng cụm
│
├── app/
│   └── streamlit_app.py          # Demo
│
├── bao_cao_ket_qua/              # Báo cáo kết quả từng phần (markdown)
│   └── 01_thu_thap_du_lieu.md
│
├── reports/
│   ├── bao_cao.pdf
│   ├── slides.pdf
│   └── demo_backup.mp4           # Video dự phòng
│
└── tests/                        # Test nhỏ cho preprocessing, metrics
```

**Quy ước chung**

- Định dạng dữ liệu: CSV UTF-8, các cột `id, title, content, url, source, date, true_label`.
- `true_label` **không** đưa vào mô hình, chỉ dùng ở bước đánh giá.
- Cố định `random_state = 42` ở mọi nơi (đặt trong `src/config.py`).
- Mỗi người làm trên branch riêng (`feat/crawl`, `feat/preprocess`, …), merge vào `main` qua Pull Request.
- API key Gemini để trong `.env`, không commit.

---

## Phần 1. Các bước thực hiện chi tiết

### Bước 0. Chốt phạm vi

- **Chủ đề:** đề yêu cầu tối thiểu 5 chủ đề. Nhóm nên lấy 6–8 chủ đề để an toàn, ví dụ Thời sự, Kinh tế, Thể thao, Giải trí, Công nghệ, Sức khỏe, Giáo dục, Pháp luật.
- **Số bài:** mỗi chủ đề tối thiểu 10 bài, tổng tối thiểu 50 bài. Nên lấy 15–20 bài mỗi chủ đề, khoảng 100–150 bài, vì quá ít bài thì kết quả phân cụm không ổn định.
- **Chuẩn chung:** thống nhất cấu trúc thư mục, định dạng file và một repo GitHub cho cả nhóm.

### Bước 1. Thu thập dữ liệu

- **Nguồn:** VnExpress, Tuổi Trẻ, Thanh Niên, Dân trí. Lấy từ nhiều báo để văn phong đa dạng hơn.
- **Công cụ:** `requests` + `BeautifulSoup`, hoặc `newspaper3k`.
- **Định dạng lưu (CSV/JSON):** `id, title, content, url, source, date, true_label`.
- Giữ nhãn chủ đề gốc (`true_label`). Nhãn này không đưa vào mô hình, chỉ dùng để đánh giá cụm ở Bước 6.
- **Lọc:** bỏ bài trùng, bài quá ngắn (dưới 150 từ), bài chỉ có video hoặc ảnh.

### Bước 2. Tiền xử lý tiếng Việt

- **Làm sạch:** bỏ HTML, URL, số, ký tự đặc biệt, rồi chuyển về chữ thường.
- **Tách từ:** dùng `underthesea` hoặc `pyvi`. Bước này bắt buộc với tiếng Việt, ví dụ "bóng đá" phải thành một từ `bóng_đá`.
- **Bỏ stopwords** tiếng Việt (dùng bộ có sẵn, thêm các từ đặc thù báo chí như "theo", "cho biết", "ảnh"…).
- **Lọc** từ quá hiếm (`min_df`) và quá phổ biến (`max_df`).

### Bước 3. Biểu diễn văn bản (cả 3 cách)

| Cách | Công cụ | Ghi chú |
|---|---|---|
| Bag-of-Words | `CountVectorizer` | Đầu vào cho LDA và LSA |
| TF-IDF | `TfidfVectorizer` (ngram 1–2) | Đầu vào chính cho LSA |
| Word Embedding | Word2Vec (gensim) tự train trên tập bài, hoặc PhoW2V / fastText cc.vi | Vector của bài = trung bình vector các từ |

### Bước 4. Tách đặc trưng: LSA hoặc LDA

- Nên chọn **LSA làm chính**, vì nó chạy được trên cả BoW lẫn TF-IDF (`TruncatedSVD`, 50–100 chiều).
- **LDA** (gensim) chỉ chạy được trên BoW. Có thể làm thêm để so sánh, đồng thời lấy luôn top từ khóa của mỗi topic.
- **Word Embedding** vốn đã là vector dày nên phân cụm trực tiếp, hoặc giảm chiều bằng PCA cho đồng bộ.
- Tổng cộng **3 cấu hình đặc trưng:** BoW+LSA, TF-IDF+LSA, Embedding.

### Bước 5. Phân cụm

- **K-Means:** `sklearn.cluster.KMeans`.
- **K-Medoids:** `sklearn_extra.cluster.KMedoids`, dùng `metric="cosine"`.
- **Chọn k:** dùng Elbow và Silhouette, đồng thời chạy với k = số chủ đề thật để so sánh.
- **Kết quả:** 3 cách biểu diễn × 2 mô hình = **6 thí nghiệm**.

| | K-Means | K-Medoids |
|---|---|---|
| BoW + LSA | TN1 | TN2 |
| TF-IDF + LSA | TN3 | TN4 |
| Embedding | TN5 | TN6 |

### Bước 6. Đánh giá và trực quan hóa

- **Chỉ số nội tại** (không cần nhãn): Silhouette, Davies-Bouldin, Calinski-Harabasz.
- **Chỉ số so với nhãn gốc:** ARI, NMI, Purity.
- Lập **bảng so sánh 6 thí nghiệm** và chọn cấu hình tốt nhất.
- **Hình minh họa:** scatter plot 2D (t-SNE/UMAP) tô màu theo cụm, wordcloud cho từng cụm, confusion matrix giữa cụm và nhãn thật.

### Bước 7. Sinh tên chủ đề cho từng cụm

- **Từ khóa:** lấy top 10–15 từ của mỗi cụm (từ centroid TF-IDF hoặc thành phần LSA).
- **spaCy:** spaCy chưa có model tiếng Việt chính thức. Dùng `spacy.blank("vi")` kèm `pyvi` để tách từ và đếm cụm danh từ; phần này chỉ mang tính bổ trợ.
- **Gemini (phần chính):** gửi prompt gồm top từ khóa và 3–5 bài gần tâm cụm nhất, yêu cầu trả về tên chủ đề ngắn và mô tả 1–2 câu.
- So sánh tên chủ đề Gemini sinh ra với nhãn thật.

### Bước 8. Demo (Streamlit)

- Cho người dùng chọn cách biểu diễn, mô hình và k, sau đó hiển thị scatter plot, wordcloud và tên chủ đề của từng cụm.
- Cho dán một bài báo mới hoặc URL, app dự đoán bài thuộc cụm nào và chủ đề gì.
- Nên chuẩn bị sẵn **video quay màn hình** làm phương án dự phòng, phòng khi mất mạng hoặc API lỗi lúc thuyết trình.

### Bước 9. Báo cáo, slide, thuyết trình

- **Báo cáo gồm:** giới thiệu, dữ liệu, phương pháp, thí nghiệm, kết quả, demo, kết luận.
- **Slide** khoảng 15–20 trang, **thuyết trình** khoảng 15 phút.

---

## Luồng dữ liệu giữa các bước

```
data/raw/*.json
   └─(merge_data)→ data/processed/articles.csv
        └─(preprocessing)→ data/processed/articles_clean.csv
             ├─ BoW ──── LSA ─┐
             ├─ TF-IDF ─ LSA ─┼─→ K-Means / K-Medoids ─→ metrics + figures ─→ outputs/tables/experiments.csv
             └─ Embedding ────┘                                  │
                                                                 └─→ keywords + Gemini ─→ outputs/topic_names/
                                                                                              └─→ app/streamlit_app.py
```
