# Báo cáo kết quả – Phần 4: Phân cụm + đánh giá

- **Người phụ trách:** Anh Dũng (TV4)
- **Script:** `src/clustering/{kmeans,kmedoids,select_k}.py`, `src/evaluation/metrics.py`
- **Notebook:** `notebooks/03_clustering_experiments.ipynb`
- **Đầu ra:** `outputs/tables/experiments.csv`, `outputs/tables/cluster_labels.csv`, `outputs/figures/elbow_silhouette_<biểu diễn>_<mô hình>.png`

## 1. Tóm tắt

| Hạng mục | Kết quả |
|---|---|
| Mô hình | K-Means (`sklearn`, chuẩn hóa L2, `n_init=10`), K-Medoids (thư viện `kmedoids`, PAM, khoảng cách cosine) |
| Chỉ số | Silhouette (cosine), Davies-Bouldin, Calinski-Harabasz, ARI, NMI, Purity |
| Số thí nghiệm đã chạy | 44 dòng trong `experiments.csv`: 5 mức số chiều LSA × 2 biểu diễn LSA + embedding, × 2 mô hình, × 2 giá trị k |
| Số chiều LSA chọn | BoW + LSA: 10, TF-IDF + LSA: 10; embedding giữ 80 (`config.W2V_SIZE`) |
| **Cấu hình tốt nhất** | **BoW + LSA (10 chiều) + K-Means, k = 6**: ARI 0,516, NMI 0,594, Purity 0,748, Silhouette 0,496 |

## 2. Cách làm

```
load_features(rep) (TV3)            rep ∈ {bow_lsa, tfidf_lsa, embedding}
  └─ quét số chiều LSA {10, 20, 30, 50, 80}   build_and_save_all_features(n_components=...)
       └─ scan_k: k = 2..12, ghi inertia (Elbow) + Silhouette → k tốt nhất = Silhouette cao nhất
            └─ chạy với k tốt nhất và k = 6 → evaluate() → experiments.csv
```

- **Chỉ số nội tại tính trên ma trận đã chuẩn hóa L2**, Silhouette dùng cosine, để K-Means (Euclid trên vector chuẩn hóa) và K-Medoids (cosine) được so trên cùng thước đo.
- `run_kmeans` trả về `(nhãn, tâm cụm, inertia)`; `run_kmedoids` trả về `(nhãn, chỉ số bài làm tâm, inertia)`.
- `contingency()` và `purity()` đã chuyển sang `src/evaluation/metrics.py`, `plots.py` import lại.

## 3. Chọn số chiều LSA

Chỉ số nội tại luôn tăng khi giảm số chiều (ít chiều thì điểm gần nhau hơn), nên không dùng để so giữa các số chiều. Chọn theo trung bình ARI và NMI ở k = 6, lấy trung bình cả hai mô hình:

| Biểu diễn | 10 | 20 | 30 | 50 | 80 |
|---|---|---|---|---|---|
| BoW + LSA | **0,551** | 0,517 | 0,496 | 0,434 | 0,482 |
| TF-IDF + LSA | **0,525** | 0,514 | 0,457 | 0,452 | 0,399 |
| Embedding | — | — | — | — | 0,504 |

10 và 20 chiều gần nhau; cả hai tốt hơn rõ 80 chiều (khớp nhận xét ở báo cáo phần 3). Số chiều chọn được đã ghi vào `config.LSA_COMPONENTS = 10`.

**Kết quả nhạy với số chiều.** Quét thêm các mức quanh 10 (K-Means, k = 6) cho thấy ARI dao động mạnh giữa các mức liền kề:

| Số chiều | 3 | 5 | 8 | 10 | 15 |
|---|---|---|---|---|---|
| BoW + LSA (ARI) | 0,155 | 0,226 | 0,395 | **0,515** | 0,510 |
| TF-IDF + LSA (ARI) | 0,376 | 0,510 | 0,447 | 0,465 | 0,511 |

Vì vậy chênh lệch vài phần trăm giữa các cấu hình ở mục 4 (ví dụ ARI 0,516 so với 0,472) nằm trong mức dao động này, chưa đủ để khẳng định cấu hình nào tốt hơn hẳn. Kết luận chắc chắn hơn là xu hướng: **ít chiều (10–20) tốt hơn rõ nhiều chiều (50–80)**.

## 4. Bảng 6 thí nghiệm (số chiều đã chọn)

| Biểu diễn | Mô hình | k | Silhouette | DB ↓ | CH | ARI | NMI | Purity |
|---|---|---|---|---|---|---|---|---|
| BoW + LSA | K-Means | **6** | 0,496 | 1,298 | 32,3 | **0,516** | 0,594 | **0,748** |
| BoW + LSA | K-Means | 11 | 0,580 | 0,977 | 34,8 | 0,479 | 0,592 | 0,765 |
| BoW + LSA | K-Medoids | 6 | 0,478 | 1,313 | 30,8 | 0,508 | 0,585 | 0,706 |
| BoW + LSA | K-Medoids | 10 | 0,585 | 0,993 | 35,5 | 0,494 | 0,598 | 0,765 |
| TF-IDF + LSA | K-Means | 6 | 0,487 | 1,199 | 30,5 | 0,466 | 0,597 | 0,681 |
| TF-IDF + LSA | K-Means | 10 | 0,649 | 0,785 | 42,6 | 0,414 | 0,583 | 0,740 |
| TF-IDF + LSA | K-Medoids | 6 | 0,471 | 1,166 | 27,8 | 0,464 | 0,572 | 0,706 |
| TF-IDF + LSA | K-Medoids | 10 | 0,649 | 0,771 | 41,6 | 0,361 | 0,560 | 0,689 |
| Embedding | K-Means | 6 | 0,289 | 1,887 | 15,5 | 0,472 | **0,605** | 0,681 |
| Embedding | K-Means | 12 | 0,307 | 1,658 | 11,4 | 0,409 | 0,600 | 0,740 |
| Embedding | K-Medoids | 6 | 0,282 | 1,832 | 15,1 | 0,391 | 0,546 | 0,639 |
| Embedding | K-Medoids | 10 | 0,288 | 1,667 | 11,8 | 0,424 | 0,576 | 0,706 |

Nhận xét:

- **K-Means ≥ K-Medoids** ở hầu hết cấu hình; chênh lệch nhỏ trên LSA, rõ hơn trên embedding.
- **k theo Silhouette luôn rơi vào 10–12**, sát cuối khoảng quét. Đường Elbow không có điểm gãy rõ (gãy nhẹ quanh k = 6–7). Chọn k lớn làm Silhouette tăng nhưng tách chủ đề thành nhiều cụm nhỏ, ARI giảm.
- **Embedding** có NMI cao nhất nhưng chỉ số nội tại kém nhất (Silhouette ≈ 0,29): các cụm chồng lấn nhiều.

## 5. Cấu hình tốt nhất: BoW + LSA (10 chiều) + K-Means, k = 6

Xét cùng lúc ba mặt, không chỉ ARI/NMI vì nhãn gốc có nhiễu:

- **So với nhãn:** ARI và Purity cao nhất trong các thí nghiệm k = 6, NMI chỉ kém embedding 0,01.
- **Nội tại:** Silhouette 0,496, DB 1,30, tốt hơn hẳn embedding.
- **Dễ hiểu:** mỗi cụm có một chủ đề chiếm đa số, đặt tên được ngay:

| Cụm | Công nghệ | Giải trí | Kinh tế | Sức khỏe | Thể thao | Thời sự | Chủ đề chính |
|---|---|---|---|---|---|---|---|
| 0 | **12** | 1 | 6 | 0 | 1 | 0 | Công nghệ (+ Kinh tế công nghệ) |
| 1 | 1 | **18** | 2 | 0 | 0 | 0 | Giải trí |
| 2 | 0 | 0 | 0 | **18** | 1 | 3 | Sức khỏe |
| 3 | 0 | 0 | 1 | 0 | **17** | 1 | Thể thao |
| 4 | 1 | 0 | **9** | 0 | 0 | 1 | Kinh tế |
| 5 | 6 | 1 | 2 | 2 | 0 | **15** | Thời sự |

Lỗi chủ yếu: Kinh tế bị chia đôi (6 bài sang cụm Công nghệ, đúng nhóm dễ lẫn Công nghệ ↔ Kinh tế ở báo cáo phần 2) và 6 bài Công nghệ rơi vào cụm Thời sự.

**Hạn chế của cách chọn.** Số chiều LSA và k = 6 được chọn có dùng nhãn thật (ARI/NMI, số chủ đề thật), nên ARI/NMI của cấu hình tốt nhất có phần lạc quan so với một bài toán phân cụm hoàn toàn không có nhãn. Nếu chỉ dựa vào chỉ số nội tại, Silhouette sẽ chọn k = 10–12 (mục 4).

## 6. Bàn giao

| Đầu ra | Cách dùng |
|---|---|
| `outputs/tables/experiments.csv` | Mỗi dòng 1 thí nghiệm: `representation, n_components, model, k` + 6 chỉ số (cả 5 mức số chiều) |
| `outputs/tables/cluster_labels.csv` | `id, representation, n_components, model, k, cluster` cho 12 thí nghiệm ở mục 4. Cấu hình tốt nhất: lọc `representation == "bow_lsa"`, `model == "kmeans"`, `k == 6` |
| Tâm cụm / bài làm tâm (TV5) | `X = load_features("bow_lsa")` (10 chiều theo `config`) → `run_kmeans(X, 6)` trả về `(nhãn, tâm cụm, inertia)`; K-Medoids: `run_kmedoids(X, 6)[1]` là chỉ số bài làm tâm |

**Số chiều:** `config.LSA_COMPONENTS = 10`. `load_features(name)` luôn trả về đúng số chiều này: nếu file trong `models/` sai kích thước (ví dụ còn bản 80 chiều cũ, hoặc vừa quét số chiều khác trong notebook 03) thì tự tạo lại. Muốn số chiều khác thì truyền rõ `load_features(name, n_components=20)`.

## 7. Chạy lại

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/03_clustering_experiments.ipynb   # ~30 giây
```
