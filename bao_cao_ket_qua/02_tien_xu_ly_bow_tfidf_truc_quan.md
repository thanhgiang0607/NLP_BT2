# Báo cáo kết quả – Phần 2: Tiền xử lý, EDA, BoW/TF-IDF, trực quan

- **Người phụ trách:** Thanh Hằng (TV2)
- **Script:** `src/preprocessing/{clean,tokenizer,stopwords,build_clean}.py`, `src/representation/{bow_tfidf,tokens}.py`, `src/visualization/plots.py`
- **Notebook:** `notebooks/01_eda.ipynb`
- **Đầu ra:** `data/processed/articles_clean.csv`, `models/{bow,tfidf}_{matrix.npz,vectorizer.joblib}`, `outputs/figures/*.png`

## 1. Tóm tắt

| Hạng mục | Kết quả |
|---|---|
| Số bài xử lý | 119 (6 chủ đề) |
| Công cụ tách từ | pyvi (có tuỳ chọn underthesea) |
| Stopwords | 293 từ (`data/stopwords/vietnamese_stopwords.txt`) |
| Token sau tiền xử lý | 37.501 token / 6.207 từ vựng; trung bình 315 token/bài (102–1.131) |
| Ma trận BoW (min_df=2, max_df=0,8, unigram) | 119 × 2.823, thưa 94,7% |
| Ma trận TF-IDF (min_df=2, max_df=0,8, ngram 1–2, sublinear_tf) | 119 × 5.089, thưa 96,2% |

## 2. Quy trình tiền xử lý

```
title + content
  └─ clean_text(lowercase=False): NFC, bỏ HTML/URL/email, bỏ số, bỏ ký tự đặc biệt
       └─ tokenize (pyvi): "Việt Nam" -> Việt_Nam
            └─ hạ chữ thường từng token
                 └─ bỏ stopwords + token < 2 ký tự -> cột `tokens` trong articles_clean.csv
```

Quyết định đáng chú ý:

- **Hạ chữ thường SAU khi tách từ.** Nếu hạ trước, pyvi mất tín hiệu chữ hoa của tên riêng: "việt nam" bị tách thành `việt` + `nam` (lần chạy đầu `nam` xuất hiện 217 lần, `việt` 186 lần). Sửa lại thì `việt_nam` là một từ (180 lần).
- **Ghép `title` + `content`** vì tiêu đề chứa từ khóa chủ đề rất mạnh.
- **Stopwords bổ sung theo dữ liệu thật:** sau lần chạy đầu, các từ như `người`, `việc`, `sự`, `khiến`, `khoảng`, `một_số` xuất hiện ở hầu hết chủ đề nên được thêm vào danh sách.
- **BoW dùng unigram, TF-IDF dùng ngram (1, 2):** với ~120 bài, bigram làm BoW quá thưa; TF-IDF dùng bigram theo kế hoạch nhóm và bắt được cụm có nghĩa (`jensen + huang`, `mạng + xã_hội`).
- **`min_df=2`:** từ vựng giảm 6.207 → 2.823 (min_df=3: 1.951; min_df=5: 1.153). Có 2.762 từ chỉ xuất hiện 1 lần.
- Vectorizer fit trên toàn bộ tập bài (không có train/test). Bài mới trong demo chỉ `.transform()`.

## 3. Kết quả EDA

- Dữ liệu cân bằng: 19–20 bài/chủ đề, 4 nguồn × ~30 bài.
- Độ dài (token sau tiền xử lý): Thể thao ngắn nhất (TB 264), Kinh tế dài nhất (TB 349); Thời sự và Sức khỏe có bài dài bất thường (max 1.131 và 922).
- 64% từ vựng là từ ghép (có `_`), nên tách từ có ảnh hưởng lớn tới chất lượng đặc trưng.
- Top từ TF-IDF mỗi chủ đề khớp trực giác (Thể thao: `hlv, trận, đội_tuyển, bóng_đá`; Sức khỏe: `bác_sĩ, bệnh_viện, điều_trị`; Thời sự: `tỉnh, ubnd, quốc_hội`).
- **Chủ đề dễ lẫn:** Công nghệ ↔ Kinh tế (cùng có `nvidia`, `công_ty`, `doanh_nghiệp`); Thời sự ↔ các chủ đề còn lại do nhãn theo chuyên mục báo có nhiễu (xem báo cáo thu thập dữ liệu, mục 7).

## 4. Hình

| File (`outputs/figures/`) | Nội dung |
|---|---|
| `eda_topic_counts.png` | Số bài theo chủ đề |
| `eda_length_raw.png`, `eda_length_tokens.png` | Độ dài bài trước và sau tiền xử lý |
| `eda_top_words.png` | Top 10 từ theo chủ đề |
| `eda_zipf.png` | Phân bố tần suất từ (Zipf) |
| `scatter_tsne_true.png`, `scatter_umap_true.png` | t-SNE / UMAP trên TF-IDF, tô màu theo chủ đề thật |
| `wordcloud_true.png` | Wordcloud mỗi chủ đề (tần suất × idf) |
| `cm_demo.png` | Minh họa confusion matrix (K-Means k=6, chỉ để kiểm tra hàm vẽ, không phải kết quả chính thức) |

Trên t-SNE, Thể thao, Sức khỏe và Giải trí tách khá rõ; Công nghệ, Kinh tế, Thời sự chồng lấn nhiều.

## 5. Hạn chế

- pyvi vẫn tách chưa hoàn hảo: `man` và `utd` rời nhau (Man Utd), `tỉ` + `đồng` rời nhau, `bộ` và `trưởng` đôi khi rời nhau. Ảnh hưởng nhỏ ở quy mô 119 bài. Có thể thử `--backend underthesea` để so sánh.
- Stopwords xây tay + theo tần suất trên đúng bộ dữ liệu này; dữ liệu mới có thể cần bổ sung.
- Dữ liệu tập trung vào một ngày nên có từ mang tính sự kiện (`nvidia`, `huang`, `utd`) trội lên.

## 6. Cách dùng cho các phần sau

```python
from src.representation.bow_tfidf import load_corpus, build_bow, build_tfidf, load_matrix
df, docs = load_corpus()                 # df.true_label chỉ dùng để đánh giá
X_bow, vec_bow = load_matrix("bow")      # hoặc build_bow(docs)
X_tfidf, vec_tfidf = load_matrix("tfidf")
tokens_list = [d.split() for d in docs]  # cho Word2Vec

from src.visualization import plots     # với nhãn cụm của cấu hình tốt nhất
plots.plot_scatter_2d(X_lsa, clusters, "tsne", title="...", name="scatter_best")
plots.plot_wordcloud_grid(docs, clusters, name="wordcloud_best", weights=plots.idf_weights(vec_tfidf))
plots.plot_confusion_matrix(df.true_label, clusters, name="cm_best")
```

## 7. Cách tái lập

```bash
python -m src.preprocessing.build_clean   # -> data/processed/articles_clean.csv
python -m src.representation.bow_tfidf    # -> models/
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb   # -> outputs/figures/
python -m tests.test_preprocessing
```
