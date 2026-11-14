## Bước 4: Biểu diễn Word Embedding & Trích xuất đặc trưng (LSA / LDA)

### 1. Mục tiêu và Phương pháp thực hiện
* **Word Embedding (`src/representation/embedding.py`):** Huấn luyện mô hình **Word2Vec (Skip-gram)** trực tiếp trên tập dữ liệu tin tức đã tách từ (`tokens`) với `vector_size = 80`, `window = 5`, `min_count = 2`, `epochs = 30`, cố định `random_state = 42` (`workers = 1` để đảm bảo tái lập 100%). Vector biểu diễn mỗi bài báo được tính bằng trung bình cộng vector của các từ có trong từ điển.
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

#### b. Kiểm tra chất lượng ngữ nghĩa Word2Vec
Mô hình Word2Vec học được mối liên hệ ngữ cảnh rõ rệt giữa các từ ghép tiếng Việt trong tập tin tức:
* **`bóng_đá`**: *thường_niên* (0.79), *olympic* (0.75), *quyền_hạn* (0.75), *cơ_cấu* (0.74), *suất* (0.73)
* **`công_nghệ`**: *khoa_học* (0.73), *lõi* (0.71), *vắc* (0.70), *uav* (0.70), *công_đoạn* (0.69)
* **`kinh_tế`**: *vĩ_mô* (0.73), *martin_sommer* (0.71), *tăng_trưởng* (0.68), *củng_cố* (0.64), *nhìn_nhận* (0.64)
* **`bệnh_viện`**: *đa_khoa* (0.85), *tân_hưng* (0.83), *khánh_hội* (0.83), *chủ_lực* (0.82), *thực_trạng* (0.82)
* **`apple`**: *trang_web* (0.84), *vĩnh_viễn* (0.81), *dạng* (0.81), *diễn_đàn* (0.81), *đóng_cửa* (0.80)

#### c. Top 10 từ khóa theo 6 chủ đề đối chiếu từ LDA
* **Topic 0 (Giải trí / Thể thao):** `việt_nam`, `phim`, `đội_tuyển`, `diễn_viên`, `trận`, `hlv`, `nghệ_sĩ`, `giải`, `phương_thanh`, `trận_đấu`
* **Topic 1 (Sức khỏe):** `bệnh_viện`, `bác_sĩ`, `bệnh`, `tăng`, `điều_trị`, `bệnh_nhân`, `khám`, `thuốc`, `y_tế`, `người_bệnh`
* **Topic 2 (Kinh tế):** `nvidia`, `nhà`, `huang`, `giá`, `thuế`, `đồng`, `doanh_nghiệp`, `tiền`, `tài_chính`, `đầu_tư`
* **Topic 3 (Thời sự):** `lực_lượng`, `gas`, `lớn`, `việt_nam`, `tuổi`, `nhà`, `đường`, `khu_vực`, `nổ`, `vụ`
* **Topic 4 (Thể thao / An ninh mạng):** `hlv`, `utd`, `man`, `lừa_đảo`, `dùng`, `cảnh_báo`, `đội`, `man_utd`, `carrick`, `clb`
* **Topic 5 (Công nghệ):** `việt_nam`, `bộ`, `công_nghệ`, `microsoft`, `tỉnh`, `quản_lý`, `dữ_liệu`, `hoạt_động`, `cơ_sở`, `tổ_chức`

---

### 3. Hướng dẫn chạy và nạp đặc trưng cho Bước 5 (Phân cụm)

#### Chạy pipeline tạo đặc trưng
```bash
# Đảm bảo đã có ma trận BoW và TF-IDF từ Bước 3
python -m src.representation.bow_tfidf

# Huấn luyện Word2Vec và trích xuất toàn bộ 3 ma trận đặc trưng vào models/
python -m src.representation.embedding
python -m src.features.lsa
```

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

---

### Lưu ý cuối trước khi Push & Tạo Pull Request
1. Kiểm tra lại biến `LSA_COMPONENTS` trong `src/config.py`: nếu bạn đã chạy thực tế ở mức **`80` chiều** như kết quả trên, hãy cập nhật `LSA_COMPONENTS = 80` trong `src/config.py` để đồng bộ với code của bạn.
2. Kiểm tra file `requirements.txt` đã có dòng `gensim==<version>` chưa.
3. Chạy `git add .` → `git commit -m "feat(features): hoan thien Word2Vec, LSA/LDA va bao cao buoc 4"` → `git push -u origin feat/features`, sau đó vào GitHub tạo Pull Request để merge vào `main`.