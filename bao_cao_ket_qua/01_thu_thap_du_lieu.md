# Báo cáo kết quả – Phần 1: Thu thập & gộp dữ liệu

- **Người phụ trách:** Thanh Giang (TV1)
- **Ngày thu thập:** 13/11/2026
- **Script:** `src/crawl/sources.py`, `src/crawl/crawler.py`, `src/crawl/merge_data.py`
- **Đầu ra:** `data/raw/<chu_de>.json` (6 file), `data/processed/articles.csv`

## 1. Tóm tắt


| Tiêu chí          | Yêu cầu đề bài           | Kết quả                                      | Đạt |
| ----------------- | ------------------------ | -------------------------------------------- | --- |
| Tổng số bài       | ≥ 50                     | **119**                                      | ✅   |
| Số chủ đề         | ≥ 5                      | **6**                                        | ✅   |
| Số bài mỗi chủ đề | ≥ 10                     | **19–20**                                    | ✅   |
| Số nguồn báo      | –                        | 4 (VnExpress, Tuổi Trẻ, Thanh Niên, Dân trí) | ✅   |
| Độ dài tối thiểu  | ≥ 150 từ (kế hoạch nhóm) | ngắn nhất 226 từ                             | ✅   |




## 2. Quy trình

```
Trang chuyên mục (6 chủ đề × 4 báo)
  └─ lấy link bài theo regex của từng báo           → 1.074 link ứng viên
       └─ tải từng bài, kiểm tra breadcrumb (JSON-LD) → đúng chuyên mục mới giữ
            └─ trích tiêu đề + sapo + thân bài, bỏ chú thích ảnh, tên tác giả
                 └─ lọc < 150 từ, bài video/ảnh       → data/raw/*.json (120 bài)
                      └─ merge_data: chuẩn hóa, lọc trùng → data/processed/articles.csv (119 bài)
```

- Mỗi chủ đề lấy 20 bài, chia đều 5 bài/báo; báo nào thiếu thì báo khác bù.
- `true_label` lấy theo chuyên mục của báo. Chỉ dùng để đánh giá, **không** đưa vào mô hình.
- `id` được đánh lại thành `doc_0001` … `doc_0119` để không lộ nhãn.



### Ánh xạ chuyên mục


| Chủ đề    | VnExpress                   | Tuổi Trẻ   | Thanh Niên | Dân trí    |
| --------- | --------------------------- | ---------- | ---------- | ---------- |
| Thời sự   | thoi-su                     | thoi-su    | thoi-su    | xa-hoi     |
| Kinh tế   | kinh-doanh                  | kinh-doanh | kinh-te    | kinh-doanh |
| Thể thao  | the-thao                    | the-thao   | the-thao   | the-thao   |
| Giải trí  | giai-tri                    | giai-tri   | giai-tri   | giai-tri   |
| Công nghệ | so-hoa (Khoa học công nghệ) | cong-nghe  | cong-nghe  | cong-nghe  |
| Sức khỏe  | suc-khoe                    | suc-khoe   | suc-khoe   | suc-khoe   |




## 3. Kết quả crawl (lần chạy cuối)


| Chủ đề    | Link ứng viên | Bài bị bỏ khi crawl | Bài lưu vào `data/raw` |
| --------- | ------------- | ------------------- | ---------------------- |
| Thời sự   | 170           | 4                   | 20                     |
| Kinh tế   | 163           | 0                   | 20                     |
| Thể thao  | 192           | 0                   | 20                     |
| Giải trí  | 171           | 3                   | 20                     |
| Công nghệ | 173           | 5                   | 20                     |
| Sức khỏe  | 205           | 0                   | 20                     |
| **Tổng**  | **1.074**     | **12**              | **120**                |


Lý do bỏ bài khi crawl:


| Lý do                                                        | Số bài |
| ------------------------------------------------------------ | ------ |
| Sai chuyên mục (link "đọc nhiều"/"liên quan" thuộc mục khác) | 9      |
| Không có thân bài (bài video / ảnh / magazine)               | 3      |
| Dưới 150 từ                                                  | 0      |




## 4. Kết quả gộp & lọc (`merge_data.py`)


| Bước lọc                           | Số bài bị bỏ  |
| ---------------------------------- | ------------- |
| Thiếu tiêu đề / nội dung           | 0             |
| Dưới 150 từ                        | 0             |
| Trùng URL                          | 0             |
| Trùng tiêu đề                      | 0             |
| Gần trùng nội dung (Jaccard ≥ 0,8) | 1             |
| **Còn lại**                        | **119 / 120** |


Bài bị loại: *"Chủ tịch FIFA nói xúc động về World Cup có U.17 Việt Nam"* (Thanh Niên) trùng 83% từ vựng với *"Chủ tịch FIFA nói gì về World Cup lần đầu có Việt Nam tham dự?"* (Tuổi Trẻ). Hai báo cùng đăng lại một bản tin.

## 5. Thống kê bộ dữ liệu cuối



### Phân bố chủ đề × nguồn


| Chủ đề    | Dân trí | Thanh Niên | Tuổi Trẻ | VnExpress | Tổng    |
| --------- | ------- | ---------- | -------- | --------- | ------- |
| Công nghệ | 5       | 5          | 5        | 5         | 20      |
| Giải trí  | 5       | 5          | 5        | 5         | 20      |
| Kinh tế   | 5       | 5          | 5        | 5         | 20      |
| Sức khỏe  | 5       | 5          | 5        | 5         | 20      |
| Thể thao  | 5       | 4          | 5        | 5         | 19      |
| Thời sự   | 5       | 5          | 5        | 5         | 20      |
| **Tổng**  | **30**  | **29**     | **30**   | **30**    | **119** |




### Độ dài bài (số từ, tách theo khoảng trắng, chưa tách từ tiếng Việt)


| Chủ đề      | Ngắn nhất | Trung bình | Trung vị | Dài nhất  | Số đoạn TB |
| ----------- | --------- | ---------- | -------- | --------- | ---------- |
| Công nghệ   | 293       | 761        | 626      | 2.073     | 15         |
| Giải trí    | 512       | 782        | 696      | 1.380     | 14         |
| Kinh tế     | 227       | 782        | 720      | 1.555     | 14         |
| Sức khỏe    | 291       | 724        | 652      | 1.980     | 15         |
| Thể thao    | 285       | 616        | 613      | 956       | 11         |
| Thời sự     | 226       | 711        | 520      | 2.995     | 14         |
| **Toàn bộ** | **226**   | **730**    | **676**  | **2.995** | –          |


- Tổng số từ: **86.891**.
- Theo nguồn (TB số từ/bài): Tuổi Trẻ 809, Dân trí 805, VnExpress 666, Thanh Niên 638.

Phân bố độ dài:


| Khoảng số từ | 150–300 | 301–500 | 501–800 | 801–1.200 | 1.201–3.000 |
| ------------ | ------- | ------- | ------- | --------- | ----------- |
| Số bài       | 5       | 23      | 55      | 27        | 9           |




### Thời gian đăng

Từ 09/11/2026 đến 13/11/2026, trong đó **108/119 bài (91%) đăng ngày 13/11**; 8 bài ngày 12/11, 2 bài ngày 11/11, 1 bài ngày 09/11.

### Ví dụ tiêu đề


| Chủ đề    | Nguồn     | Tiêu đề                                                                            |
| --------- | --------- | ---------------------------------------------------------------------------------- |
| Công nghệ | VnExpress | Vì sao sự cố Microsoft kéo dài bất thường?                                         |
| Công nghệ | Dân trí   | Đề xuất mạng lưới cảnh báo lừa đảo trực tuyến tại Việt Nam                         |
| Giải trí  | Dân trí   | Phương Thanh - Anh Dũng: Cặp vợ chồng duy nhất được truy tặng NSND                 |
| Giải trí  | VnExpress | Phim AI chuyển thể 'Liêu trai chí dị' được khen                                    |
| Kinh tế   | Dân trí   | Giá vàng thế giới bất ngờ tăng nhẹ sau khi chạm đáy 13 tháng                       |
| Kinh tế   | VnExpress | IMF đánh giá Việt Nam kiểm soát tốt rủi ro lạm phát                                |
| Sức khỏe  | VnExpress | 5 đồ uống giàu vitamin C giúp tăng cường miễn dịch                                 |
| Sức khỏe  | VnExpress | Bác sĩ gắp hàng trăm nọc độc ong trên cơ thể người phụ nữ                          |
| Thể thao  | Dân trí   | Bruno Fernandes từ chối gia hạn hợp đồng với Man Utd                               |
| Thể thao  | Tuổi Trẻ  | 'Địa chấn' môn cờ tướng, kỳ thủ Việt Nam 18 tuổi giành HCV cá nhân ở giải thế giới |
| Thời sự   | VnExpress | Hai nguyên lãnh đạo Bình Phước 'vi phạm gây hậu quả rất nghiêm trọng'              |
| Thời sự   | Dân trí   | Tây Ninh gấp rút xử lý 477 trụ sở dôi dư trước ngày 15/12                          |




## 6. Vấn đề gặp phải & cách xử lý


| Vấn đề                                                                                                | Cách xử lý                                                     |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Trang chuyên mục chứa link của mục khác (sidebar, "đọc nhiều")                                        | Kiểm tra breadcrumb trong JSON-LD của từng bài, sai mục thì bỏ |
| VnExpress đổi tên mục "Số hóa" thành "Khoa học công nghệ" → lần chạy đầu 0 bài Công nghệ từ VnExpress | Bổ sung tên mục mới vào `ACCEPTED_SECTIONS`                    |
| VnExpress: sapo nằm trong thân bài → bị lặp                                                           | Tách sapo ra khỏi thân bài trước khi trích                     |
| Bài ảnh (photo story) lặp lại chú thích                                                               | Bỏ đoạn văn trùng trong cùng bài                               |
| Các báo đăng lại cùng một bản tin                                                                     | Lọc gần trùng bằng Jaccard trên tập từ                         |




## 7. Lưu ý cho các phần sau

- **Dữ liệu tập trung vào 1 ngày** (91% ngày 13/11) → nhiều bài cùng sự kiện (nhiều bài Man Utd đổi HLV, vụ nổ gas TP.HCM). Các bài cùng sự kiện có thể tạo cụm con theo sự kiện thay vì theo chủ đề.
- **Một sự kiện trải qua nhiều chủ đề:** 7 bài về Jensen Huang rời Nvidia nằm ở 3 nhãn khác nhau (Công nghệ 3, Kinh tế 3, Giải trí 1). Nhiều khả năng mô hình gom chúng vào cùng một cụm, làm giảm ARI/NMI dù cụm đó hợp lý về nội dung.
- **Nhãn theo chuyên mục báo có nhiễu:** vd. Dân trí xếp bài "Vụ nổ gas ở TPHCM" vào Sức khỏe; cả Dân trí và Tuổi Trẻ xếp bài phát biểu của Phó Bí thư Thành ủy TP.HCM vào Sức khỏe. Nên nhắc khi phân tích ARI/NMI/Purity.
- **Chủ đề dễ lẫn:** Thời sự ↔ Kinh tế ↔ Sức khỏe (tin xã hội), Công nghệ ↔ Kinh tế (tin doanh nghiệp công nghệ).
- `content` gồm sapo + thân bài, các đoạn ngăn cách bằng `\n`; đã chuẩn hóa Unicode NFC.
- Muốn tăng độ đa dạng: chạy lại crawler vào ngày khác rồi chạy `merge_data` (tự lọc trùng).



## 8. Cách tái lập (nhưng cũng không cần tái lập)

```bash
.venv/bin/python -m src.crawl.crawler      # 20 bài/chủ đề -> data/raw/
.venv/bin/python -m src.crawl.merge_data   # -> data/processed/articles.csv
```

> Lưu ý: chạy lại sẽ lấy tin mới nhất tại thời điểm chạy nên nội dung sẽ khác bộ dữ liệu trong báo cáo này.

