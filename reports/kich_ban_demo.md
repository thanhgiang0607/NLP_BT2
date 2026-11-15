# Kịch bản demo – Phân cụm & sinh tên chủ đề tin tức tiếng Việt

- **Thời lượng demo:** khoảng 4–5 phút (trong 15 phút thuyết trình)
- **Người thao tác máy:** TV5 (Xuân Hoàng) · **Người dẫn lời:** _điền tên_
- **Cấu hình chính:** BoW + LSA (10 chiều) + K-Means, k = 6 · tên chủ đề do Gemini sinh, lưu sẵn trong `outputs/topic_names/bow_lsa_kmeans_k6.json`

> Tất cả kết quả mong đợi trong kịch bản đã được chạy thử qua app ngày 15/11/2026.

---

## 0. Chuẩn bị trước buổi thuyết trình

### Trước 1 ngày

- [ ] `git pull` bản mới nhất trên `main`.
- [ ] Chạy `.venv/bin/python -m src.pipeline` → tổng kết cuối phải có `khop_cluster_labels_csv: True` và `ten_chu_de: khớp`, không có dòng `CẢNH BÁO`.
- [ ] Chạy thử toàn bộ kịch bản này một lần, kiểm tra kết quả khớp mục "Kết quả mong đợi".
- [ ] **Chọn lại URL** ở bước 3: URL trong kịch bản là bài ngày 15/11; nếu đến hôm thuyết trình bài đã cũ, lấy 3 bài mới cùng chuyên mục và thử trước.
- [ ] Quay video dự phòng `reports/demo_backup.mp4` theo đúng kịch bản (TV1).

### Trước 15 phút

```bash
cd NLP_BT2
.venv/bin/python app/streamlit_app.py      # Windows: python app/streamlit_app.py
```

- [ ] App mở ở `http://localhost:8501`, chờ bảng 6 cụm hiện đủ (lần mở đầu khoảng 5 giây).
- [ ] Mở sẵn một file ghi chú chứa 4 đoạn văn bản và 4 URL ở mục 3–4 để copy nhanh.
- [ ] Kiểm tra mạng (phần URL cần mạng). Tắt thông báo hệ thống, phóng to trình duyệt (Ctrl/Cmd +) cho người xem dễ đọc.
- [ ] Mở sẵn video dự phòng ở một tab khác.

---

## 1. Mở đầu (30 giây)

**Thao tác:** đang ở tab **Khám phá cụm**, để nguyên cấu hình mặc định.

**Lời dẫn:**
> "Đây là ứng dụng demo của nhóm. Dữ liệu là 119 bài báo từ 4 báo VnExpress, Tuổi Trẻ, Thanh Niên, Dân trí thuộc 6 chuyên mục. Mô hình **không được biết nhãn chuyên mục**, nó tự gom bài thành cụm, sau đó Gemini đọc từ khóa và vài bài tiêu biểu của mỗi cụm để đặt tên chủ đề."

---

## 2. Tab 1 – Khám phá cụm (1,5 phút)

### 2.1. Bảng tên chủ đề (cấu hình tốt nhất)

**Thao tác:** giữ **BoW + LSA · K-Means · k = 6**, chỉ vào bảng.

**Kết quả mong đợi:**

| Cụm | Số bài | Tên chủ đề (Gemini) | Chuyên mục chiếm đa số |
|---|---|---|---|
| 0 | 20 | Công nghệ toàn cầu | Công nghệ (60%) |
| 1 | 21 | Giải trí nghệ thuật | Giải trí (86%) |
| 2 | 22 | Y tế sức khỏe | Sức khỏe (82%) |
| 3 | 19 | Bóng đá Việt Nam | Thể thao (89%) |
| 4 | 11 | Kinh tế thị trường | Kinh tế (82%) |
| 5 | 26 | Quản lý nhà nước | Thời sự (58%) |

**Lời dẫn:**
> "Đây là cấu hình tốt nhất trong 6 thí nghiệm, đạt ARI 0,52, NMI 0,59, Purity 0,75. Gemini không được cho biết 6 nhãn gốc, nhưng 4 trên 6 cụm được đặt tên gần trùng chuyên mục. Hai cụm còn lại được đặt theo nội dung thực tế: cụm 3 hầu hết là bóng đá nên tên hẹp hơn 'Thể thao', còn cụm 5 trộn tin thời sự với tin chính sách công nghệ nên Gemini gọi là 'Quản lý nhà nước'."

### 2.2. Scatter t-SNE và wordcloud

**Thao tác:** cuộn xuống biểu đồ t-SNE, rồi wordcloud.

**Lời dẫn:**
> "Mỗi chấm là một bài báo, tô màu theo cụm. Thể thao, Sức khỏe, Giải trí tách khá rõ; Công nghệ, Kinh tế, Thời sự chồng lấn nhau, đúng với nhận xét ở phần đánh giá. Wordcloud cho thấy từ khóa đặc trưng của từng cụm, ví dụ cụm Y tế có *bệnh viện, bác sĩ, điều trị*."

### 2.3. Đổi cấu hình (tùy thời gian)

**Thao tác:** đổi **Mô hình → K-Medoids**, hoặc kéo **k → 8**.

**Lời dẫn:**
> "App cho chọn 3 cách biểu diễn, 2 mô hình và k từ 2 đến 12. Với các cấu hình ngoài cấu hình chính, tên cụm được ghép tạm từ 3 từ khóa hàng đầu, vì nhóm chỉ gọi Gemini cho cấu hình tốt nhất để app chạy được khi không có mạng."

**Thao tác:** trả lại **K-Means, k = 6** trước khi sang tab 2.

---

## 3. Tab 2 – Dự đoán bài mới bằng văn bản (1 phút)

**Thao tác:** chuyển sang tab **Dự đoán bài mới** → chọn **"Dán nội dung văn bản"** → dán từng đoạn → bấm **Dự đoán chủ đề**.

**Lời dẫn:**
> "Bài mới đi qua đúng các bước như lúc huấn luyện: làm sạch, tách từ, BoW, LSA 10 chiều, rồi gán vào cụm có tâm gần nhất."

| # | Đoạn văn bản (copy nguyên) | Kết quả mong đợi |
|---|---|---|
| T1 | Giá vàng hôm nay tăng mạnh. Giá vàng miếng SJC tăng 2 triệu đồng mỗi lượng, ngân hàng nhà nước điều chỉnh tỷ giá, nhà đầu tư lo ngại lạm phát và lãi suất. | Cụm 4 – Kinh tế thị trường |
| T2 | Bệnh viện Bạch Mai vừa tiếp nhận nhiều bệnh nhân sốt xuất huyết nặng. Bác sĩ khuyến cáo người dân không tự ý dùng thuốc hạ sốt, cần đến cơ sở y tế để được khám và điều trị kịp thời. | Cụm 2 – Y tế sức khỏe |
| T3 | Bộ phim mới của đạo diễn Victor Vũ thu hút đông đảo khán giả ngay trong ngày công chiếu. Nam diễn viên chính chia sẻ anh đã giảm 10 kg để vào vai, nhiều nghệ sĩ có mặt tại buổi ra mắt. | Cụm 1 – Giải trí nghệ thuật |
| T4 | Đội tuyển Việt Nam thắng Malaysia 2-0 trong trận đấu thuộc vòng loại Asian Cup. HLV Kim Sang-sik hài lòng với tinh thần thi đấu của các cầu thủ. | Cụm 3 – Bóng đá Việt Nam |

Nếu thiếu thời gian, chỉ demo **T1** và **T2**.

---

## 4. Tab 2 – Dự đoán bài mới bằng URL (1,5 phút)

**Thao tác:** chọn **"Nhập URL bài báo"** → dán URL → chờ dòng "Đã trích … từ từ URL" → bấm **Dự đoán chủ đề**.

**Lời dẫn:**
> "Đây là các bài đăng hôm nay, **không có trong 119 bài huấn luyện**. App tự tải trang và bóc tách tiêu đề, tóm tắt, thân bài."

| # | Bài (chuyên mục của báo) | URL | Kết quả mong đợi |
|---|---|---|---|
| U1 | Thủ môn ném bóng về lưới nhà ở Argentina (Thể thao, VnExpress) | https://vnexpress.net/thu-mon-nem-bong-ve-luoi-nha-o-argentina-5144841.html | ✅ Cụm 3 – Bóng đá Việt Nam |
| U2 | Trà thảo mộc mang lại lợi ích sức khỏe như thế nào? (Sức khỏe, Tuổi Trẻ) | https://tuoitre.vn/tra-thao-moc-mang-lai-loi-ich-suc-khoe-nhu-the-nao-100261115170152356.htm | ✅ Cụm 2 – Y tế sức khỏe |
| U3 | 'Trại buôn người' có suất chiếu đặc biệt tại Nhật sau khi thu gần 500 tỉ (Giải trí, Thanh Niên) | https://thanhnien.vn/trai-buon-nguoi-co-suat-chieu-dac-biet-tai-nhat-sau-khi-thu-gan-500-ti-185261115185604662.htm | ✅ Cụm 1 – Giải trí nghệ thuật |
| U4 | Nhà đầu tư Hàn Quốc ồ ạt bắt đáy cổ phiếu Nvidia sau vụ bê bối của CEO (Kinh tế, Dân trí) | https://dantri.com.vn/kinh-doanh/nha-dau-tu-han-quoc-o-at-bat-day-co-phieu-nvidia-sau-vu-be-boi-cua-ceo-20261115214455021.htm | 🟡 Cụm 0 – Công nghệ toàn cầu |

**Lời dẫn cho U4 (điểm nhấn về hạn chế):**
> "Bài này báo xếp vào Kinh tế, nhưng mô hình đưa vào cụm Công nghệ toàn cầu. Lý do là cụm 0 vốn gom các tin doanh nghiệp công nghệ, trong đó có Nvidia. Mô hình nhóm theo **nội dung**, nên với các chủ đề giao nhau như kinh tế công nghệ, kết quả có thể khác cách báo phân chuyên mục. Đây cũng là cặp dễ lẫn nhất mà nhóm đã chỉ ra ở phần đánh giá."

**Không dùng khi demo:** bài thời tiết https://tuoitre.vn/tphcm-mua-som-nhung-khong-lon-100261115063209109.htm bị xếp nhầm vào Cụm 2 – Y tế sức khỏe, vì dữ liệu huấn luyện không có tin thời tiết. Chỉ đưa ra nếu được hỏi về trường hợp mô hình sai.

---

## 5. Kết thúc demo (20 giây)

**Lời dẫn:**
> "Tóm lại, app cho phép khám phá 6 thí nghiệm phân cụm, xem tên chủ đề do Gemini sinh và dự đoán chủ đề cho bài báo mới từ văn bản hoặc URL. Hạn chế chính là dữ liệu nhỏ, 119 bài trong khoảng một ngày, nên các chủ đề giao nhau và chủ đề chưa gặp như thời tiết còn dự đoán chưa tốt."

---

## 6. Xử lý sự cố

| Sự cố | Cách xử lý |
|---|---|
| Mất mạng / URL không tải được (app báo "Không trích được nội dung") | Bỏ qua mục 4, demo bằng văn bản ở mục 3. Tab 1 chạy hoàn toàn không cần mạng |
| Bài trong URL đã bị gỡ hoặc đổi | Dùng URL dự phòng đã chuẩn bị, hoặc copy nội dung bài dán vào chế độ văn bản |
| App không mở được / báo lỗi | Chạy lại `.venv/bin/python -m src.pipeline` rồi mở lại app; nếu vẫn lỗi, chiếu video `reports/demo_backup.mp4` |
| App chậm sau mỗi lần bấm (khoảng 1,5 giây là bình thường) | Chờ, không bấm liên tục. Mỗi lần bấm app vẽ lại t-SNE và wordcloud ở tab 1 |
| Kết quả khác kịch bản | Kiểm tra cấu hình tab 1 đang là BoW + LSA · K-Means · k = 6; tab 2 luôn dùng cấu hình này |

---

## 7. Câu hỏi có thể gặp

| Câu hỏi | Gợi ý trả lời |
|---|---|
| Tại sao chọn k = 6? | Silhouette chọn k = 10–12 (sát mép khoảng quét) nhưng tách chủ đề thành nhiều cụm nhỏ, ARI giảm. k = 6 bằng số chuyên mục và cho ARI tốt nhất. Nhóm thừa nhận việc chọn k và số chiều có dùng nhãn thật nên ARI có phần lạc quan (báo cáo phần 4) |
| Tại sao LSA chỉ 10 chiều? | Với 119 bài, LSA tối đa 118 chiều; 80 chiều gần như không giảm chiều và giữ nhiễu. Quét {10, 20, 30, 50, 80} thì 10–20 chiều tốt hơn rõ. Kết quả khá nhạy với số chiều nên kết luận chắc chắn là xu hướng "ít chiều tốt hơn" |
| Gemini có "gian lận" bằng cách biết nhãn không? | Không. Prompt chỉ có 15 từ khóa và tiêu đề + đoạn đầu của 5 bài gần tâm cụm, không có danh sách 6 chuyên mục |
| Tại sao K-Means tốt hơn K-Medoids? | Chênh lệch nhỏ trên LSA (ARI 0,516 so với 0,508), rõ hơn trên embedding (0,472 so với 0,391). Một giải thích có thể: tâm cụm trung bình của K-Means đại diện cho cả cụm, còn K-Medoids buộc phải lấy một bài cụ thể làm tâm, dễ lệch khi cụm trộn nhiều chủ đề |
| Word2Vec có tốt không? | Dữ liệu chỉ khoảng 37 nghìn token nên quan hệ giữa các từ còn nhiễu, nhưng vector trung bình của bài vẫn tách chủ đề khá tốt (NMI cao nhất, 0,605) |
| Mô hình sai ở đâu? | Kinh tế ↔ Công nghệ (tin doanh nghiệp công nghệ), Thời sự ↔ Công nghệ (tin chính sách), và chủ đề chưa có trong dữ liệu như thời tiết |
