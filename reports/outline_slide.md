# Outline slide – Phân cụm & sinh tên chủ đề tin tức tiếng Việt

- **Thời lượng:** 15 phút thuyết trình (gồm khoảng 4–5 phút demo) · **18 slide chính + 4 slide dự phòng**
- **Nguồn số liệu:** `bao_cao_ket_qua/01–05`, `outputs/tables/*.csv`, `outputs/figures/*.png`, `reports/kich_ban_demo.md`
- **Quy ước:** mỗi slide gồm **Mục tiêu** (người xem cần nhớ gì), **Trên slide** (chữ, bảng, hình đặt lên), **Lời nói** (gợi ý nói), **Nguồn**.
- **Người trình bày:** đề xuất mỗi thành viên trình bày phần mình làm, điền tên vào cột "Người nói" ở bảng phân bổ.

## Phân bổ thời gian

| Phần | Slide | Người nói | Thời gian |
|---|---|---|---|
| Mở đầu, bài toán | 1–3 | TV1 | 1,5 phút |
| Dữ liệu | 4–5 | TV1 | 1,5 phút |
| Tiền xử lý, biểu diễn | 6–8 | TV2, TV3 | 2 phút |
| Phân cụm, đánh giá | 9–13 | TV4 | 3 phút |
| Sinh tên chủ đề | 14–15 | TV5 | 1,5 phút |
| Demo | 16 | TV5 thao tác, 1 người dẫn | 4 phút |
| Kết luận | 17–18 | TV1 | 1 phút |

## Cần chuẩn bị thêm trước khi làm slide

Ba hình theo **cụm của cấu hình tốt nhất** đã có trong `outputs/figures/`. Vẽ lại bằng `python -m src.visualization.best_figures` (nhãn cụm lấy từ `cluster_labels.csv`, tên cụm từ JSON của Gemini, mỗi cụm cùng một màu ở mọi hình):

| Hình | Nội dung | Dùng ở slide |
|---|---|---|
| `scatter_tsne_best.png` | t-SNE trên BoW + LSA 10 chiều, tô màu theo cụm, chú thích "Cụm i · tên Gemini (số bài)" | 13 |
| `cm_best.png` | Confusion matrix cụm × chủ đề thật | 12 |
| `wordcloud_best.png` | Wordcloud mỗi cụm, tiêu đề là tên Gemini | 14 |

Còn thiếu: **ảnh chụp màn hình app** (tab 1 và tab 2) cho slide 16, dự phòng nếu demo lỗi.

**Không dùng** `cm_demo.png`: đây chỉ là hình minh họa kiểm tra hàm vẽ (TF-IDF + SVD), không phải kết quả chính thức.

---

## Slide 1 – Trang bìa

- **Trên slide:** tên đề tài "Phân cụm và sinh tên chủ đề cho tin tức tiếng Việt"; môn học, giảng viên; tên 5 thành viên kèm vai trò (TV1 Thanh Giang – dữ liệu & tích hợp, TV2 Thanh Hằng – tiền xử lý & trực quan, TV3 Gia Khang – đặc trưng, TV4 Anh Dũng – phân cụm & đánh giá, TV5 Xuân Hoàng – sinh chủ đề & demo); link repo GitHub.
- **Lời nói:** "Nhóm em trình bày đề tài phân cụm tin tức tiếng Việt và tự động đặt tên chủ đề cho từng cụm."

## Slide 2 – Bài toán & yêu cầu đề bài

- **Mục tiêu:** người xem biết đề bài yêu cầu gì và nhóm đã đáp ứng đủ.
- **Trên slide:** bảng 2 cột "Yêu cầu" / "Nhóm thực hiện":

  | Yêu cầu | Nhóm thực hiện |
  |---|---|
  | ≥ 50 bài, ≥ 5 chủ đề, ≥ 10 bài/chủ đề | 119 bài, 6 chủ đề, 19–20 bài/chủ đề |
  | LSA hoặc LDA | LSA (chính) + LDA (đối chiếu) |
  | BoW, TF-IDF, Word Embedding | Cả 3 |
  | K-Means, K-Medoids | Cả 2 → 6 thí nghiệm |
  | Sinh chủ đề (spaCy / LLM) | Gemini (chính) + spaCy (bổ trợ) |
  | Demo | App Streamlit |

- **Lời nói:** "Bài toán là cho một tập bài báo không có nhãn, tự gom thành nhóm cùng chủ đề rồi đặt tên cho mỗi nhóm. Nhãn chuyên mục gốc chỉ dùng để chấm điểm, không đưa vào mô hình."

## Slide 3 – Tổng quan quy trình

- **Mục tiêu:** một hình nhìn là hiểu toàn bộ pipeline.
- **Trên slide:** sơ đồ ngang 6 khối, dưới mỗi khối ghi người phụ trách:
  `Thu thập (TV1)` → `Tiền xử lý (TV2)` → `Biểu diễn + LSA (TV2, TV3)` → `Phân cụm (TV4)` → `Đánh giá (TV4)` → `Sinh tên chủ đề + Demo (TV5)`.
  Góc slide: "Chạy toàn bộ bằng 1 lệnh: `python -m src.pipeline` (~7 giây)".
- **Lời nói:** "Toàn bộ quy trình tái lập được: chạy lại pipeline cho ra đúng các con số trong báo cáo."
- **Nguồn:** `KE_HOACH.md`, `src/pipeline.py`.

## Slide 4 – Thu thập dữ liệu

- **Mục tiêu:** dữ liệu đủ, đa dạng nguồn, sạch.
- **Trên slide:**
  - Nguồn: VnExpress, Tuổi Trẻ, Thanh Niên, Dân trí · công cụ `requests` + `BeautifulSoup`.
  - Phễu số liệu: **1.074 link** ứng viên → bỏ 12 bài (9 sai chuyên mục, 3 bài chỉ có video/ảnh) → **120 bài** → bỏ 1 bài gần trùng (2 báo đăng lại cùng bản tin) → **119 bài**.
  - Bảng chủ đề × nguồn: mỗi ô 5 bài (Thể thao – Thanh Niên 4).
  - Độ dài: trung bình 730 từ/bài, ngắn nhất 226, dài nhất 2.995.
- **Hình:** `eda_topic_counts.png`.
- **Lời nói:** "Để nhãn sạch, nhóm kiểm tra breadcrumb của từng bài: link 'đọc nhiều' thuộc chuyên mục khác lọt vào trang danh sách sẽ bị loại."
- **Nguồn:** `bao_cao_ket_qua/01_thu_thap_du_lieu.md`.

## Slide 5 – Đặc điểm & hạn chế của dữ liệu

- **Mục tiêu:** chuẩn bị cho người xem hiểu vì sao kết quả không hoàn hảo.
- **Trên slide:** 3 gạch đầu dòng, mỗi dòng 1 ví dụ:
  - **Tập trung 1 ngày:** 91% bài đăng ngày 13/11 → nhiều bài cùng sự kiện.
  - **Một sự kiện trải nhiều chủ đề:** 7 bài về Jensen Huang rời Nvidia nằm ở Công nghệ (3), Kinh tế (3), Giải trí (1).
  - **Nhãn của báo có nhiễu:** bài "Vụ nổ gas ở TPHCM" được Dân trí xếp vào Sức khỏe.
- **Lời nói:** "Nhãn là chuyên mục do báo đặt, không phải sự thật tuyệt đối. Điểm ARI/NMI vì vậy có trần thấp hơn 1 dù mô hình gom hợp lý."
- **Nguồn:** báo cáo 01 mục 7.

## Slide 6 – Tiền xử lý tiếng Việt

- **Mục tiêu:** cho thấy nhóm xử lý đúng đặc thù tiếng Việt.
- **Trên slide:**
  - Quy trình: chuẩn hóa NFC, bỏ HTML/URL → **tách từ bằng pyvi** → bỏ dấu câu, số, hạ chữ thường → bỏ stopwords (259 từ, có từ đặc thù báo chí).
  - Ví dụ minh họa 1 câu trước/sau: "Đội tuyển Việt Nam thắng trận bóng đá." → `đội_tuyển`, `việt_nam`, `thắng`, `trận`, `bóng_đá`.
  - Kết quả: 37.267 token, 5.994 từ vựng, trung bình 313 token/bài; **64% từ vựng là từ ghép**.
- **Điểm nhấn (1 dòng đóng khung):** "Tách từ **trước** khi bỏ dấu câu: nếu làm ngược, pyvi nối nhầm `hồng_quân_phó_cục_trưởng`, `u_việt_nam`."
- **Hình:** `eda_top_words.png` (hoặc `eda_length_tokens.png`).
- **Nguồn:** báo cáo 02.

## Slide 7 – Ba cách biểu diễn văn bản

- **Mục tiêu:** phân biệt 3 cách biểu diễn.
- **Trên slide:** bảng

  | Cách | Công cụ | Kích thước |
  |---|---|---|
  | Bag-of-Words | `CountVectorizer`, `min_df=2`, `max_df=0,8` | 119 × 2.785 (thưa 94,7%) |
  | TF-IDF | `TfidfVectorizer`, ngram 1–2, `sublinear_tf` | 119 × 5.060 (thưa 96,2%) |
  | Word2Vec | Skip-gram tự train, 80 chiều, vector bài = trung bình vector từ | 119 × 80 (từ điển 3.406 từ) |

- **Hình nhỏ:** `scatter_tsne_true.png`. Thể thao, Sức khỏe, Giải trí tách rõ; Công nghệ, Kinh tế, Thời sự chồng lấn.
- **Lời nói:** "Ngay khi chưa phân cụm, chỉ tô màu theo nhãn thật, ta đã thấy 3 chủ đề dễ và 3 chủ đề khó tách."
- **Nguồn:** báo cáo 02, 03.

## Slide 8 – Tách đặc trưng: LSA (và LDA đối chiếu)

- **Mục tiêu:** vì sao chọn LSA, LDA cho thấy gì.
- **Trên slide:**
  - LSA = `TruncatedSVD` trên BoW và TF-IDF → không gian dày, ít chiều; chạy được trên cả BoW lẫn TF-IDF.
  - LDA (gensim, 6 topic) chỉ để đối chiếu: Topic "Sức khỏe" rất sạch (`bệnh_viện, bác_sĩ, điều_trị`), nhưng nhiều topic bị trộn (Giải trí + Thể thao; Công nghệ + Thời sự).
  - → 3 cấu hình đặc trưng đưa vào phân cụm: **BoW + LSA, TF-IDF + LSA, Word2Vec**.
- **Hình:** `lsa_explained_variance.png`.
- **Lời nói:** "Ban đầu nhóm chọn 80 chiều vì giữ trên 80% phương sai, nhưng phần đánh giá cho thấy đó không phải tiêu chí tốt cho phân cụm. Slide sau sẽ nói rõ."
- **Nguồn:** báo cáo 03.

## Slide 9 – Thiết kế thí nghiệm phân cụm

- **Mục tiêu:** người xem hiểu 6 thí nghiệm và cách chấm điểm.
- **Trên slide:**
  - Lưới 3 × 2: {BoW + LSA, TF-IDF + LSA, Word2Vec} × {K-Means, K-Medoids} = **6 thí nghiệm**.
  - K-Means: chuẩn hóa L2, `n_init=10`. K-Medoids: PAM trên khoảng cách cosine (thư viện `kmedoids`).
  - Quét thêm: số chiều LSA {10, 20, 30, 50, 80}, k từ 2 đến 12 → tổng 44 lần chạy.
  - 6 chỉ số: nội tại (Silhouette, Davies-Bouldin, Calinski-Harabasz) và so với nhãn (ARI, NMI, Purity).
- **Lời nói:** "Chỉ số nội tại đo cụm có gọn và tách nhau không, chỉ số so với nhãn đo cụm có khớp chuyên mục báo không."
- **Nguồn:** báo cáo 04 mục 1–2.

## Slide 10 – Chọn số chiều LSA

- **Mục tiêu:** phát hiện quan trọng: ít chiều tốt hơn nhiều chiều.
- **Trên slide:** biểu đồ đường (vẽ từ `experiments.csv`): trục x = số chiều {10, 20, 30, 50, 80}, trục y = trung bình (ARI + NMI)/2 ở k = 6, 2 đường BoW + LSA và TF-IDF + LSA:

  | | 10 | 20 | 30 | 50 | 80 |
  |---|---|---|---|---|---|
  | BoW + LSA | **0,551** | 0,517 | 0,496 | 0,434 | 0,482 |
  | TF-IDF + LSA | **0,525** | 0,514 | 0,457 | 0,452 | 0,399 |

- **Điểm nhấn:** "119 bài → LSA tối đa 118 chiều; 80 chiều gần như không giảm chiều, vẫn giữ nhiễu. Chốt **10 chiều**."
- **Lời nói (thẳng thắn):** "Kết quả khá nhạy: BoW + LSA 8 chiều chỉ đạt ARI 0,395, 10 chiều đạt 0,515. Kết luận chắc chắn là **xu hướng**: 10–20 chiều tốt hơn rõ 50–80 chiều."
- **Nguồn:** báo cáo 04 mục 3.

## Slide 11 – Chọn k

- **Mục tiêu:** vì sao dùng k = 6.
- **Trên slide:**
  - Hình `elbow_silhouette_bow_lsa_kmeans.png`.
  - 2 ý: Silhouette luôn chọn k = 10–12 (sát mép khoảng quét); Elbow không có điểm gãy rõ, gãy nhẹ quanh k = 6–7.
  - Ví dụ: BoW + LSA K-Means k = 11 có Silhouette 0,580 (cao hơn) nhưng ARI 0,479 (thấp hơn k = 6: 0,516) vì chủ đề bị chẻ thành cụm nhỏ.
- **Lời nói:** "Nhóm báo cáo cả hai: k theo Silhouette và k = 6 bằng số chuyên mục. Chọn k = 6 có dùng thông tin số nhãn, nên điểm ARI có phần lạc quan; nhóm ghi rõ hạn chế này."
- **Nguồn:** báo cáo 04 mục 4–5.

## Slide 12 – Kết quả 6 thí nghiệm & cấu hình tốt nhất

- **Mục tiêu:** bảng kết quả chính, cấu hình thắng.
- **Trên slide (bảng chính, k = 6, LSA 10 chiều):**

  | Biểu diễn | Mô hình | Silhouette | ARI | NMI | Purity |
  |---|---|---|---|---|---|
  | **BoW + LSA** | **K-Means** | **0,496** | **0,516** | 0,594 | **0,748** |
  | BoW + LSA | K-Medoids | 0,478 | 0,508 | 0,585 | 0,706 |
  | TF-IDF + LSA | K-Means | 0,487 | 0,466 | 0,597 | 0,681 |
  | TF-IDF + LSA | K-Medoids | 0,471 | 0,464 | 0,572 | 0,706 |
  | Word2Vec | K-Means | 0,289 | 0,472 | **0,605** | 0,681 |
  | Word2Vec | K-Medoids | 0,282 | 0,391 | 0,546 | 0,639 |

  Tô đậm dòng tốt nhất. Bên phải: hình `cm_best.png`.
- **3 nhận xét (chữ to, ngắn):** K-Means ≥ K-Medoids; Word2Vec NMI cao nhất nhưng cụm chồng lấn (Silhouette 0,29); lỗi chính: Kinh tế bị chia đôi, 6 bài sang cụm Công nghệ.
- **Lời nói:** "Cấu hình tốt nhất là BoW + LSA 10 chiều + K-Means, xét cả 3 mặt: khớp nhãn, cụm gọn và dễ đặt tên. Chênh lệch giữa các cấu hình chỉ vài phần trăm, nằm trong mức dao động khi đổi số chiều."
- **Nguồn:** `experiments.csv`, báo cáo 04 mục 4–5.

## Slide 13 – Các cụm tìm được

- **Mục tiêu:** cho thấy cụm có nghĩa.
- **Trên slide:** bảng chéo cụm × chủ đề thật (cấu hình tốt nhất), tô đậm ô lớn nhất mỗi hàng:

  | Cụm | Công nghệ | Giải trí | Kinh tế | Sức khỏe | Thể thao | Thời sự |
  |---|---|---|---|---|---|---|
  | 0 | **12** | 1 | 6 | 0 | 1 | 0 |
  | 1 | 1 | **18** | 2 | 0 | 0 | 0 |
  | 2 | 0 | 0 | 0 | **18** | 1 | 3 |
  | 3 | 0 | 0 | 1 | 0 | **17** | 1 |
  | 4 | 1 | 0 | **9** | 0 | 0 | 1 |
  | 5 | 6 | 1 | 2 | 2 | 0 | **15** |

  Bên cạnh: `scatter_tsne_best.png`.
- **Lời nói:** "Mỗi cụm có một chủ đề chiếm đa số. Cụm 0 gom tin doanh nghiệp công nghệ nên kéo theo 6 bài Kinh tế về Nvidia, Microsoft; cụm 5 gom tin chính sách nên lẫn bài Công nghệ về chuyển đổi số."
- **Nguồn:** báo cáo 04 mục 5.

## Slide 14 – Sinh tên chủ đề bằng Gemini

- **Mục tiêu:** quy trình đặt tên và đảm bảo khách quan.
- **Trên slide:**
  - Sơ đồ: mỗi cụm → **15 từ khóa TF-IDF** + **5 bài gần tâm cụm** (tiêu đề + ~120 từ đầu) → Gemini → JSON `{name, description}`.
  - Khung nổi bật: "Prompt **không** chứa danh sách 6 chuyên mục, nên Gemini không thể chép lại nhãn."
  - spaCy (`spacy.blank("vi")` + pyvi) đếm từ ghép để đối chiếu, ví dụ cụm Thể thao: `đội_tuyển, trận_đấu, asian_cup, bóng_đá`.
  - Kết quả lưu sẵn → app chạy được không cần mạng.
- **Hình:** `wordcloud_best.png`, tiêu đề mỗi ô là tên Gemini.
- **Nguồn:** báo cáo 05 mục 2–3.

## Slide 15 – Tên chủ đề so với nhãn thật

- **Mục tiêu:** đánh giá chất lượng tên.
- **Trên slide:**

  | Cụm | Tên Gemini | Chủ đề đa số | Tỉ lệ |
  |---|---|---|---|
  | 0 | Công nghệ toàn cầu | Công nghệ | 60% |
  | 1 | Giải trí nghệ thuật | Giải trí | 86% |
  | 2 | Y tế sức khỏe | Sức khỏe | 82% |
  | 3 | Bóng đá Việt Nam | Thể thao | 89% |
  | 4 | Kinh tế thị trường | Kinh tế | 82% |
  | 5 | Quản lý nhà nước | Thời sự | 58% |

- **Lời nói:** "4 trên 6 cụm được đặt tên gần trùng chuyên mục. 2 cụm còn lại Gemini đặt theo nội dung thực tế: cụm 3 phần lớn là bóng đá nên tên hẹp hơn 'Thể thao'; cụm 5 trộn tin thời sự với tin chính sách nên thành 'Quản lý nhà nước'. Tên phản ánh đúng việc cụm bị trộn."
- **Nguồn:** `topic_vs_label.csv`, báo cáo 05 mục 4.

## Slide 16 – Demo

- **Mục tiêu:** chuyển sang app trực tiếp.
- **Trên slide:** tiêu đề "Demo", 2 ảnh chụp nhỏ (tab Khám phá cụm, tab Dự đoán bài mới), lệnh chạy `python app/streamlit_app.py`.
- **Thao tác:** làm theo `reports/kich_ban_demo.md`, mục 1–5 (khoảng 4 phút):
  1. Tab 1: bảng 6 tên chủ đề → t-SNE → wordcloud → đổi K-Medoids / k = 8.
  2. Tab 2 – văn bản: giá vàng → *Kinh tế thị trường*; bệnh viện → *Y tế sức khỏe*.
  3. Tab 2 – URL bài mới: thể thao, sức khỏe, giải trí (đúng); cổ phiếu Nvidia → *Công nghệ toàn cầu* (minh họa chủ đề giao nhau).
- **Dự phòng:** mất mạng thì chỉ demo văn bản; app lỗi thì chiếu `reports/demo_backup.mp4`.

## Slide 17 – Kết luận & hạn chế

- **Mục tiêu:** tổng kết điều đã làm được và nhìn nhận hạn chế.
- **Trên slide:** 2 cột.
  - **Đạt được:** 119 bài, 6 chủ đề, 3 biểu diễn × 2 mô hình; cấu hình tốt nhất ARI 0,52, NMI 0,59, Purity 0,75; Gemini đặt tên đúng 4/6 cụm mà không biết nhãn; app demo dự đoán bài mới; pipeline tái lập bằng 1 lệnh.
  - **Hạn chế:** dữ liệu nhỏ, tập trung 1 ngày; nhãn báo có nhiễu; chọn k và số chiều có dùng nhãn nên điểm có phần lạc quan; chủ đề chưa gặp (ví dụ thời tiết) dự đoán sai; Word2Vec thiếu dữ liệu để học tốt.
- **Lời nói:** chọn 1 câu chốt: "Với dữ liệu nhỏ, biểu diễn đơn giản như BoW + LSA ít chiều lại hiệu quả nhất."

## Slide 18 – Hướng phát triển & cảm ơn

- **Trên slide:**
  - Thu thập nhiều ngày hơn, nhiều chủ đề hơn (pipeline đã có `--crawl`).
  - Dùng embedding huấn luyện sẵn (PhoW2V, PhoBERT) thay Word2Vec tự train.
  - Chọn k không cần nhãn (gap statistic, ổn định cụm).
  - Cho Gemini đặt tên cho mọi cấu hình, không chỉ cấu hình tốt nhất.
  - "Cảm ơn thầy/cô và các bạn đã lắng nghe" + link repo.

---

## Slide dự phòng (để sau slide 18, chỉ mở khi được hỏi)

| Slide | Nội dung | Nguồn |
|---|---|---|
| D1 – Toàn bộ 44 thí nghiệm | Bảng đầy đủ `experiments.csv` (mọi số chiều, mọi k) | `outputs/tables/experiments.csv` |
| D2 – Độ nhạy số chiều | Bảng ARI theo 3, 5, 8, 10, 15 chiều cho BoW + LSA và TF-IDF + LSA | báo cáo 04 mục 3 |
| D3 – Word2Vec | Láng giềng gần nhất (hợp lý: `bệnh_viện – đa_khoa`; nhiễu: `bóng_đá – thường_niên`) và lý do (chỉ ~37 nghìn token) | báo cáo 03 mục 2b |
| D4 – Prompt Gemini | Nguyên văn prompt trong `src/topic_naming/gemini_namer.py` | code |

## Câu hỏi thường gặp

Dùng bảng "Câu hỏi có thể gặp" ở cuối `reports/kich_ban_demo.md` (vì sao k = 6, vì sao 10 chiều, Gemini có biết nhãn không, K-Means so với K-Medoids, Word2Vec, mô hình sai ở đâu).

## Gợi ý thiết kế

- Mỗi slide tối đa 3–4 ý; số liệu chính để **chữ to** (ARI 0,52, 119 bài, 6 cụm…).
- Dùng **cùng một màu cho mỗi chủ đề** ở mọi hình và bảng (bảng màu `PALETTE` trong `plots.py`), để người xem nhận ra chủ đề xuyên suốt.
- Bảng số dùng dấu phẩy thập phân thống nhất (0,516), tô đậm giá trị tốt nhất.
- Hình chèn từ `outputs/figures/` (bản 200 dpi) để không vỡ khi chiếu.
