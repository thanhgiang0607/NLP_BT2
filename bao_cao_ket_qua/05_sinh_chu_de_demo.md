# Báo cáo kết quả – Phần 5: Sinh tên chủ đề & Demo

- **Người phụ trách:** Xuân Hoàng (TV5)
- **Script:** `src/topic_naming/{keywords,spacy_np,gemini_namer}.py`, `app/streamlit_app.py`
- **Notebook:** `notebooks/04_topic_naming.ipynb`
- **Đầu ra:** `outputs/topic_names/bow_lsa_kmeans_k6.json`, `outputs/tables/topic_vs_label.csv`, ứng dụng demo Streamlit

---

## 1. Tóm tắt

| Hạng mục | Kết quả |
|---|---|
| Cấu hình tốt nhất (từ Phần 4) | **BoW + LSA (`10` chiều) + K-Means, `k = 6`** (`config.LSA_COMPONENTS = 10`) |
| Trích xuất từ khóa & bài tiêu biểu | Top 15 từ khóa TF-IDF theo trung bình cụm; Top 5 bài viết gần tâm cụm nhất theo cosine (tiêu đề + 120 từ đầu) |
| Phân tích bổ trợ spaCy | `spacy.blank("vi")` + `pyvi` (tái sử dụng văn bản đã nối từ ghép bằng `_`) để đếm top 10 từ ghép mỗi cụm |
| Sinh tên & mô tả bằng Gemini | Gọi Gemini (`gemini-3.5-flash`, đầu ra JSON, không đưa 6 nhãn thật vào prompt), lưu sẵn vào `outputs/topic_names/bow_lsa_kmeans_k6.json` |
| Đối chiếu nhãn thật (`topic_vs_label.csv`) | 6 cụm đều khớp rõ một chủ đề đa số với tỷ lệ từ **57,69%** (*Thời sự*) đến **89,47%** (*Thể thao*) |
| Demo Streamlit (`app/streamlit_app.py`) | 2 tab: **Khám phá cụm** (chọn biểu diễn, mô hình, `k`, bảng tên chủ đề, scatter t-SNE, wordcloud) và **Dự đoán bài mới** (văn bản hoặc URL) |

---

## 2. Quy trình sinh tên chủ đề

```
load_features("bow_lsa") + run_kmeans(X, k=6)
  ├─ top_keywords(X_tfidf, v_tfidf, labels, n=15)   → top 15 từ khóa TF-IDF mỗi cụm
  ├─ nearest_articles(X, labels, centers, df, n=5)  → 5 bài gần tâm nhất (cosine trên vector chuẩn hóa L2)
  ├─ top_compound_nouns(docs, labels, n=10)         → 10 từ ghép spaCy phổ biến nhất (bổ trợ đối chiếu)
  └─ name_cluster(keywords, articles)               → gọi Gemini sinh JSON {"name", "description"}
       └─ outputs/topic_names/bow_lsa_kmeans_k6.json + outputs/tables/topic_vs_label.csv
```

- **Giữ tính khách quan khi đối chiếu nhãn thật:** Prompt gửi Gemini chỉ chứa 15 từ khóa TF-IDF và tiêu đề + đoạn mở đầu (~120 từ) của 5 bài gần tâm nhất, **tuyệt đối không** liệt kê danh sách 6 nhãn thật (`Thời sự`, `Kinh tế`, `Thể thao`, `Giải trí`, `Công nghệ`, `Sức khỏe`) để tránh việc mô hình sao chép lại nhãn gốc.
- **Xử lý chung cho K-Means và K-Medoids:** Hàm `cluster_centers_l2()` trong `src/topic_naming/keywords.py` chuẩn hóa L2 tâm cụm 2D của K-Means và lấy trực tiếp vector đã chuẩn hóa L2 theo chỉ số medoid 1D của K-Medoids để dùng thống nhất phép đo cosine.
- **Chạy offline ổn định khi thuyết trình:** Kết quả từ Gemini cho cấu hình tốt nhất được lưu cố định dạng `{cluster: {name, description, keywords}}` trong `outputs/topic_names/bow_lsa_kmeans_k6.json`. App Streamlit đọc trực tiếp file JSON này khi chạy để không phụ thuộc mạng hay gọi lại API mỗi lần thao tác.

---

## 3. Từ khóa TF-IDF, từ ghép spaCy và bài viết gần tâm (`BoW + LSA + K-Means, k = 6`)

| Cụm | Top 10 từ khóa TF-IDF | Top 10 từ ghép spaCy (`_`) | Bài gần tâm nhất |
|---|---|---|---|
| **0** | `công_ty`, `nvidia`, `huang`, `microsoft`, `mỹ`, `hãng`, `dùng`, `ceo`, `sự_cố`, `nhà` | `công_ty`, `sự_cố`, `hoạt_động`, `dịch_vụ`, `nhân_viên`, `công_nghệ`, `ảnh_hưởng`, `dự_đoán`, `đầu_tư`, `jensen_huang` | *Nguyên nhân khiến iPhone 18 trễ hẹn tháng 9* |
| **1** | `khán_giả`, `diễn_viên`, `phim`, `nghệ_sĩ`, `nam`, `ca_sĩ`, `nữ`, `hát`, `vai`, `sân_khấu` | `diễn_viên`, `nghệ_sĩ`, `khán_giả`, `ca_sĩ`, `gia_đình`, `phương_thanh`, `sân_khấu`, `tổ_chức`, `trường_sa`, `kim_seung_woo` | *Ju Ji Hoon bình thản với mọi chỉ trích trong 'Hoàng hậu tái hôn'* |
| **2** | `bác_sĩ`, `bệnh_viện`, `điều_trị`, `người_bệnh`, `cơ_thể`, `bệnh_nhân`, `bệnh`, `tăng`, `nguy_cơ`, `y_tế` | `bệnh_viện`, `bác_sĩ`, `điều_trị`, `bệnh_nhân`, `y_tế`, `người_bệnh`, `huyết_áp`, `nguy_cơ`, `khu_vực`, `cơ_thể` | *Bác sĩ gắp hàng trăm nọc độc ong trên cơ thể người phụ nữ* |
| **3** | `trận`, `hlv`, `việt_nam`, `đội_tuyển`, `đội`, `trận_đấu`, `bóng`, `giải`, `asian_cup`, `đội bóng` | `việt_nam`, `đội_tuyển`, `trận_đấu`, `asian_cup`, `nhật_bản`, `bóng_đá`, `giao_hữu`, `thi_đấu`, `hợp_đồng`, `man_utd` | *Nhật Bản hủy trận giao hữu với tuyển Việt Nam* |
| **4** | `đồng`, `thuế`, `giá`, `tiền`, `nộp`, `tỉ đồng`, `tăng`, `tỉ`, `ngân_sách`, `thị_trường` | `ngân_sách`, `nhà_nước`, `ngân_hàng`, `doanh_nghiệp`, `lãi_suất`, `cơ_quan`, `quy_định`, `thị_trường`, `tài_chính`, `quốc_hội` | *Bỏ quỹ bình ổn xăng dầu: Công cụ nào kìm giá khi có biến động mạnh?* |
| **5** | `tỉnh`, `bộ`, `quản_lý`, `tổ_chức`, `địa_phương`, `sở`, `công_an`, `yêu_cầu`, `trách_nhiệm`, `công_nghệ` | `công_nghệ`, `tổ_chức`, `việt_nam`, `dữ_liệu`, `quản_lý`, `bộ_trưởng`, `cơ_sở`, `sáng_tạo`, `pháp_luật`, `hoạt_động` | *Bộ Chính trị: Luân chuyển cán bộ giữa ngành thanh tra với các cơ quan* |

**Nhận xét về phần bổ trợ spaCy:** Do `spacy.blank("vi")` không có bộ gán nhãn từ loại POS/NP tiếng Việt sẵn, việc lọc các token đã nối bằng `_` giúp giữ lại các danh từ ghép và thuật ngữ có nghĩa rõ hơn (ví dụ `ngân_sách`, `nhà_nước`, `lãi_suất` ở cụm 4 hoặc `bóng_đá`, `giao_hữu` ở cụm 3), bổ trợ tốt cho top TF-IDF vốn có cả một số từ đơn phổ biến (`tỉnh`, `bộ`, `đồng`, `tăng`).

---

## 4. Tên chủ đề Gemini sinh ra & Đối chiếu nhãn thật (`topic_vs_label.csv`)

| Cụm | Tên chủ đề (`gemini_name`) | Mô tả do Gemini sinh | Chủ đề đa số (`chu_de_chiem_da_so`) | Tỷ lệ (`ti_le`) |
|---|---|---|---|---|
| 0 | **Công nghệ toàn cầu** | Tổng hợp các chuyển động nổi bật của các tập đoàn công nghệ lớn như Apple, Microsoft, Amazon và Nvidia. Các bài viết đề cập đến sự cố hạ tầng, sản phẩm mới và những diễn biến xung quanh các lãnh đạo công nghệ thế giới. | Công nghệ | 0,6000 (12/20) |
| 1 | **Giải trí nghệ thuật** | Tổng hợp các tin tức nổi bật xoay quanh đời sống nghệ sĩ, phim ảnh và các chương trình nghệ thuật sân khấu trong nước và quốc tế. | Giải trí | 0,8571 (18/21) |
| 2 | **Y tế sức khỏe** | Tổng hợp các ca điều trị bệnh lý, cấp cứu ngoại khoa cùng lời khuyên y khoa từ bác sĩ chuyên khoa. Nội dung giúp người đọc nâng cao kiến thức chăm sóc sức khỏe và phòng ngừa các rủi ro y tế. | Sức khỏe | 0,8182 (18/22) |
| 3 | **Bóng đá Việt Nam** | Tổng hợp tin tức về các trận đấu giao hữu, kế hoạch chuẩn bị và các giải đấu quốc tế liên quan đến đội tuyển bóng đá Việt Nam. | Thể thao | 0,8947 (17/19) |
| 4 | **Kinh tế thị trường** | Cập nhật các biến động về giá cả hàng hóa, chính sách tài chính ngân hàng, thuế và hoạt động khuyến mãi tiêu dùng. | Kinh tế | 0,8182 (9/11) |
| 5 | **Quản lý nhà nước** | Tổng hợp các chỉ đạo, chính sách mới và hoạt động quản lý, điều hành của các cơ quan chức năng trên nhiều lĩnh vực đời sống, kinh tế và xã hội. | Thời sự | 0,5769 (15/26) |

**Nhận xét đối chiếu:**
- Dù **không nhận danh sách 6 nhãn thật**, Gemini đặt cho **4/6 cụm** (0, 1, 2, 4) tên gần như trùng chuyên mục gốc (*Công nghệ*, *Giải trí*, *Sức khỏe*, *Kinh tế*).
- 2 cụm còn lại được đặt tên theo **nội dung thực tế của cụm** thay vì tên chuyên mục, và điều này phản ánh đúng việc cụm bị trộn hoặc hẹp hơn chuyên mục:
  - Cụm **3** tên *Bóng đá Việt Nam*, hẹp hơn *Thể thao* vì đa số bài trong cụm là bóng đá, dù cụm còn có bài các môn khác (cờ tướng, bóng bàn).
  - Cụm **5** tên *Quản lý nhà nước*, không phải *Thời sự*: chủ đề đa số chỉ chiếm 57,7%, cụm còn chứa bài Công nghệ về chính sách, chuyển đổi số và bài thu hồi sản phẩm của cơ quan chức năng.
- Cụm **0 (60,0%)** ngoài 12 bài Công nghệ còn gom 6 bài Kinh tế liên quan tới tập đoàn công nghệ (Nvidia, Microsoft, Apple).
- Cụm **5 (57,69%)** mang tên *Quản lý nhà nước* rất sát thực tế vì ngoài 15 bài Thời sự còn chứa 6 bài Công nghệ về chính sách quản lý và chuyển đổi số của bộ, ngành, địa phương (khớp với phân tích ở báo cáo phần 4).

---

## 5. Ứng dụng Demo Streamlit (`app/streamlit_app.py`)

Ứng dụng gồm 2 tab theo đúng thiết kế trong `KE_HOACH.md` và `PHAN_CONG.md`:

1. **Tab 1 – Khám phá cụm:**
   - Chọn cách biểu diễn (`BoW + LSA`, `TF-IDF + LSA`, `Word Embedding`), mô hình (`K-Means`, `K-Medoids`) và số cụm `k` (`2..12`, mặc định `6`).
   - Hiển thị **bảng tên chủ đề** (`Cụm`, `Số bài`, `Tên chủ đề`, `Mô tả`, `Từ khóa`), **biểu đồ phân tán 2D (t-SNE)** (`plots.plot_scatter_2d`) và **Wordcloud từng cụm** (`plots.plot_wordcloud_grid`).
   - Cấu hình mặc định (`BoW + LSA + K-Means, k = 6`) đọc trực tiếp kết quả Gemini từ `outputs/topic_names/bow_lsa_kmeans_k6.json`.
2. **Tab 2 – Dự đoán bài mới:**
   - Sử dụng cấu hình tốt nhất đã chốt (`BoW + LSA 10 chiều + K-Means, k = 6`).
   - Hỗ trợ **dán nội dung văn bản** hoặc **nhập URL bài báo** (tải HTML bằng `crawler.fetch(url)` và bóc tách tiêu đề/sapo/nội dung theo `SELECTORS` của 4 báo trong `src/crawl/sources.py`, kèm fallback `<h1>`/`<article>` cho báo khác).
   - Văn bản đi qua đúng chuỗi xử lý lúc huấn luyện (`preprocess_text` → `bow_vectorizer` → `bow_lsa_svd` → `normalize` L2) và gán về cụm có tâm gần nhất (đã kiểm tra khớp **119/119 bài** so với `outputs/tables/cluster_labels.csv`).

> *Ghi chú trước khi nộp/thuyết trình:* Chụp màn hình Tab 1 & Tab 2 khi chạy demo để gắn vào slide, và nhờ TV1 quay video dự phòng `reports/demo_backup.mp4`.

---

## 6. Bàn giao & Chạy lại

| Đầu ra | Mục đích sử dụng |
|---|---|
| `outputs/topic_names/bow_lsa_kmeans_k6.json` | Tên chủ đề, mô tả Gemini và top 15 từ khóa cho app đọc offline |
| `outputs/tables/topic_vs_label.csv` | Bảng đối chiếu tên chủ đề Gemini với nhãn đa số từng cụm |
| `app/streamlit_app.py` | Ứng dụng demo tương tác cho buổi thuyết trình |

### Cách chạy app demo

App đã lưu sẵn kết quả tên chủ đề và tự gọi Python của `.venv` khi chạy trực tiếp (Windows: `.venv/Scripts/python.exe`, macOS / Linux: `.venv/bin/python`), nên chỉ cần **mở file `app/streamlit_app.py` rồi bấm Run** trong IDE, hoặc chạy ở thư mục gốc:

```bash
# Cách 1 (mọi hệ điều hành): tự tìm .venv rồi gọi streamlit run
python app/streamlit_app.py

# Cách 2: gọi thẳng streamlit trong .venv
.venv/bin/streamlit run app/streamlit_app.py                  # macOS / Linux
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py        # Windows (PowerShell)
```

Trình duyệt sẽ tự mở giao diện demo (không cần tạo `.env`, không cần gọi API Gemini). Lần mở đầu trên máy mới clone mất thêm vài giây để tạo đặc trưng trong `models/`.

*(Tùy chọn dành cho người muốn sinh lại tên chủ đề mới từ Gemini API: điền `GEMINI_API_KEY` vào `.env` rồi chạy `python -m src.topic_naming.gemini_namer` bằng Python của `.venv` — macOS / Linux: `.venv/bin/python -m src.topic_naming.gemini_namer`, Windows: `.\.venv\Scripts\python.exe -m src.topic_naming.gemini_namer`.)*
