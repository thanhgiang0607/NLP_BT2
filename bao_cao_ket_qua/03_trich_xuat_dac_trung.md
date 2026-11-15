# Báo cáo kết quả – Phần 3: Word Embedding & trích xuất đặc trưng (LSA / LDA)

- **Người phụ trách:** Gia Khang (TV3)
- **Script:** `src/representation/embedding.py`, `src/features/lsa.py`, `src/features/lda.py`
- **Notebook:** `notebooks/02_representation_lsa.ipynb`
- **Đầu ra:** `models/{bow_lsa,tfidf_lsa,embedding}_matrix.npy`, `models/word2vec.model`, `models/lda.model`, `outputs/figures/lsa_explained_variance.png`

### 1. Mục tiêu và Phương pháp thực hiện
* **Word Embedding (`src/representation/embedding.py`):** Huấn luyện mô hình **Word2Vec (Skip-gram)** trực tiếp trên tập dữ liệu tin tức đã tách từ (`tokens`) với `vector_size = 80` (`config.W2V_SIZE`, tách riêng khỏi số chiều LSA), `window = 5`, `min_count = 2`, `epochs = 30`, cố định `random_state = 42` (`workers = 1` để đảm bảo tái lập 100%). Vector biểu diễn mỗi bài báo được tính bằng trung bình cộng vector của các từ có trong từ điển.
* **LSA - Latent Semantic Analysis (`src/features/lsa.py`):** Áp dụng `TruncatedSVD` (`n_components = 80`, `random_state = 42`) để giảm chiều hai ma trận thưa đầu vào là **BoW** và **TF-IDF**, loại bỏ nhiễu và thu gọn chiều dữ liệu về cùng không gian `80` chiều với Word Embedding.
* **LDA - Latent Dirichlet Allocation (`src/features/lda.py`):** Huấn luyện mô hình chủ đề `gensim.models.LdaModel` (`num_topics = 6`, `passes = 15`, `random_state = 42`) trên tập từ vựng BoW nhằm đối chiếu ngữ nghĩa và trích xuất top từ khóa đại diện cho từng chủ đề tiềm ẩn.

---

### 2. Kết quả thực nghiệm & Chọn số chiều

#### a. Thống kê các ma trận biểu diễn và giảm chiều (N = 119 bài báo)
Dựa trên biểu đồ phương sai tích lũy (*Cumulative Explained Variance*) trong `notebooks/02_representation_lsa.ipynb` (`outputs/figures/lsa_explained_variance.png`), mốc **`n_components = 80`** được chọn vì cân bằng tốt giữa khả năng nén chiều (giảm từ hàng nghìn chiều xuống 80 chiều) và tỷ lệ thông tin giữ lại (> 83% cho cả hai phương pháp):

| Phương pháp biểu diễn | Kích thước đầu vào | Độ thưa (%) | Kích thước sau trích xuất | Phương sai giữ lại (Explained Variance) | File lưu trữ (`models/`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BoW + LSA** | `(119, 2785)` | 94.70% | `(119, 80)` | **92.62%** (`0.9262`) | `bow_lsa_matrix.npy` |
| **TF-IDF + LSA** | `(119, 5060)` | 96.17% | `(119, 80)` | **83.21%** (`0.8321`) | `tfidf_lsa_matrix.npy` |
| **Word2Vec Embedding** | Từ điển 3,406 từ | — | `(119, 80)` | — | `embedding_matrix.npy` |

**Lưu ý về số chiều LSA cho phân cụm.** Với 119 bài, LSA tối đa chỉ có 118 chiều, nên giữ 80 chiều gần như không giảm chiều và vẫn giữ nhiều nhiễu. Đo nhanh bằng K-Means k = 6 (trung bình 5 seed, ma trận đã chuẩn hóa L2):

| Ma trận | Silhouette (theo nhãn thật) | ARI | NMI |
| :--- | :---: | :---: | :---: |
| BoW + LSA 80 chiều | 0,087 | 0,404 | 0,546 |
| TF-IDF + LSA 80 chiều | 0,065 | 0,335 | 0,494 |
| TF-IDF + LSA 50 chiều | 0,097 | 0,377 | 0,540 |
| TF-IDF + LSA 20 chiều | **0,210** | **0,443** | **0,599** |
| Word2Vec (trung bình) | 0,199 | 0,455 | 0,606 |

Tiêu chí "giữ > 80% phương sai" phù hợp để nén dữ liệu nhưng không phù hợp để chọn số chiều cho phân cụm. Bước 5 (TV4) nên quét `n_components` ∈ {10, 20, 30, 50, 80} và chọn theo chất lượng cụm; đổi số chiều bằng `build_and_save_all_features(n_components=...)` hoặc `config.LSA_COMPONENTS`.

#### b. Kiểm tra chất lượng ngữ nghĩa Word2Vec
Tập dữ liệu chỉ có khoảng 37 nghìn token nên Word2Vec học quan hệ giữa các từ còn nhiễu: một số láng giềng hợp lý (`bệnh_viện` – `đa_khoa`, `kinh_tế` – `vĩ_mô`, `công_nghệ` – `khoa_học`), nhưng nhiều láng giềng không liên quan (`bóng_đá` – `thường_niên`, `quyền_hạn`, `cơ_cấu`). Dù vậy, vector trung bình của bài vẫn tách chủ đề tốt nhất trong 3 cách biểu diễn (bảng trên). Top 5 láng giềng gần nhất:
* **`bóng_đá`**: *thường_niên* (0.79), *olympic* (0.75), *quyền_hạn* (0.75), *cơ_cấu* (0.74), *suất* (0.73)
* **`công_nghệ`**: *khoa_học* (0.73), *lõi* (0.71), *vắc* (0.70), *uav* (0.70), *công_đoạn* (0.69)
* **`kinh_tế`**: *vĩ_mô* (0.73), *martin_sommer* (0.71), *tăng_trưởng* (0.68), *củng_cố* (0.64), *nhìn_nhận* (0.64)
* **`bệnh_viện`**: *đa_khoa* (0.85), *tân_hưng* (0.83), *khánh_hội* (0.83), *chủ_lực* (0.82), *thực_trạng* (0.82)
* **`apple`**: *trang_web* (0.84), *vĩnh_viễn* (0.81), *dạng* (0.81), *diễn_đàn* (0.81), *đóng_cửa* (0.80)

#### c. Top 10 từ khóa theo 6 chủ đề đối chiếu từ LDA
Tên trong ngoặc là gán tay theo từ khóa. LDA không tách sạch 6 chủ đề thật: Topic 0 trộn Giải trí với Thể thao, Topic 4 trộn Thể thao với tin lừa đảo trực tuyến, Topic 5 trộn Công nghệ với Thời sự (`bộ`, `tỉnh`, `quản_lý`).

* **Topic 0 (Giải trí / Thể thao):** `việt_nam`, `phim`, `đội_tuyển`, `diễn_viên`, `trận`, `hlv`, `nghệ_sĩ`, `giải`, `phương_thanh`, `trận_đấu`
* **Topic 1 (Sức khỏe):** `bệnh_viện`, `bác_sĩ`, `bệnh`, `tăng`, `điều_trị`, `bệnh_nhân`, `khám`, `thuốc`, `y_tế`, `người_bệnh`
* **Topic 2 (Kinh tế):** `nvidia`, `nhà`, `huang`, `giá`, `thuế`, `đồng`, `doanh_nghiệp`, `tiền`, `tài_chính`, `đầu_tư`
* **Topic 3 (Thời sự):** `lực_lượng`, `gas`, `lớn`, `việt_nam`, `tuổi`, `nhà`, `đường`, `khu_vực`, `nổ`, `vụ`
* **Topic 4 (Thể thao / An ninh mạng):** `hlv`, `utd`, `man`, `lừa_đảo`, `dùng`, `cảnh_báo`, `đội`, `man_utd`, `carrick`, `clb`
* **Topic 5 (Công nghệ / Thời sự):** `việt_nam`, `bộ`, `công_nghệ`, `microsoft`, `tỉnh`, `quản_lý`, `dữ_liệu`, `hoạt_động`, `cơ_sở`, `tổ_chức`

---

### 3. Hướng dẫn chạy và nạp đặc trưng cho Bước 5 (Phân cụm)

#### Chạy pipeline tạo đặc trưng
```bash
# Một lệnh tạo lại toàn bộ: BoW/TF-IDF -> BoW+LSA, TF-IDF+LSA -> Word2Vec -> models/
python -m src.features.lsa
```

Lệnh này luôn tạo lại từ `articles_clean.csv` (vài giây), không dùng lại file cũ trong `models/`, nên đặc trưng không bị lỗi thời khi tiền xử lý hoặc config thay đổi.

#### Giao diện nạp dữ liệu chuẩn (load_features)
Thành viên thực hiện Bước 5 gọi hàm load_features(name) để lấy ma trận đặc trưng:
```python
from sklearn.preprocessing import normalize
from src.features.lsa import load_features

# name nhận một trong 3 giá trị: "bow_lsa", "tfidf_lsa", "embedding"
X = load_features("tfidf_lsa")  # trả về np.ndarray dày, shape=(119, 80), dtype=float32

# Lưu ý: Ma trận trả về CHƯA chuẩn hóa L2. Cần chuẩn hóa trước khi phân cụm:
X_norm = normalize(X, norm="l2")
```
* **Thứ tự dòng:** Khớp hoàn toàn với file data/processed/articles_clean.csv (dòng i ứng với bài viết thứ i).
* **Tính toàn vẹn:** Đã kiểm tra không chứa giá trị NaN, không chứa cột true_label.
* **Máy mới clone:** `models/` không đi theo git; `load_features` tự tạo đủ 3 ma trận nếu chưa có file.
* **Tái lập:** Word2Vec dùng `workers=1` + seed cố định, chạy ở 2 tiến trình riêng cho ma trận giống hệt nhau. Khi train, gensim có thể in `Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'`; cảnh báo này không ảnh hưởng kết quả.