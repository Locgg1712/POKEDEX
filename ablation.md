# 📊 Nghiên Cứu Bóc Tách Đặc Trưng DSP & Thuật Toán ML (Ablation Study)

## 1. Giao thức Đánh giá Khoa học (Evaluation Protocol)

Tất cả các thử nghiệm được thực hiện trên cùng một phân chia dữ liệu bất biến:
- **Phương pháp chia**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` — lấy Fold 0 làm tập kiểm định (Validation Fold).
- **Group Tracking**: Khóa nhóm `group = f"{class_name}/{filename}"` cho mỗi ảnh gốc. Toàn bộ 15 biến thể tăng cường dữ liệu (Augmentations) của một ảnh gốc bắt buộc cùng nằm ở Train hoặc cùng nằm ở Val.
- **Tiêu chuẩn chống rò rỉ (Zero Data Leakage)**: `StandardScaler` chỉ được `fit` trên tập Train, sau đó dùng để `transform` tập Val.

---

## 2. Bảng Kết Quả Thử Nghiệm Bóc Tách (Ablation Table)

| Mã | Cấu hình Thử nghiệm | Mô tả Chi tiết Đặc trưng / Mô hình | Số chiều | Top-1 Acc | Top-3 Acc | Top-5 Acc | Macro F1 |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **E00** | Baseline (DSP Cổ điển) | HOG (1764) + HSV Hist (1024) + Fourier 1D (32) | 2,820 | *đo trên Kaggle* | *đo* | *đo* | *đo* |
| **E01** | + Fine HOG | Thêm HOG thang đo mịn (cell 16x16 / 8x8) | 3,144 | *đo* | *đo* | *đo* | *đo* |
| **E02** | + Spatial Color Hist | Biểu đồ màu cục bộ theo lưới 3x3 | 3,432 | *đo* | *đo* | *đo* | *đo* |
| **E03** | + Lab Color Moments | Mean, std, skewness của 3 kênh Lab | 3,441 | *đo* | *đo* | *đo* | *đo* |
| **E04** | + LBP Texture | Local Binary Pattern 2x2 grid (kết cấu vi mô) | 3,481 | *đo* | *đo* | *đo* | *đo* |
| **E05** | + Shape & Hu Moments | Tỷ lệ diện tích, độ phức tạp lồi, 7 bất biến Hu | 3,498 | *đo* | *đo* | *đo* | *đo* |
| **E06** | **+ GLCM Texture (MỚI)** | Ma trận đồng xuất mức xám: contrast, energy, homogeneity... | 3,538 | *đo* | *đo* | *đo* | *đo* |
| **E07** | **+ Gabor Filter Bank (MỚI)**| Bộ lọc Gabor 3 tần số, 6 hướng (đáp ứng tần số có hướng) | 3,574 | *đo* | *đo* | *đo* | *đo* |
| **E08** | **+ Fourier 2D Spectrum (MỚI)**| DFT 2 chiều: năng lượng vành bán kính & nan hoa góc | 3,590 | *đo* | *đo* | *đo* | *đo* |
| **E09** | **+ DeepMLP (3 Layers)** | Mạng sâu 2048 -> 1024 -> 512, kích hoạt GELU, Dropout 0.3 | 3,590 | *đo* | *đo* | *đo* | *đo* |
| **E10** | **+ MLP Ensemble (5 Seeds)** | Kết hợp 5 hạt giống ngẫu nhiên (Ensemble Soft Voting) | 3,590 | *đo* | *đo* | *đo* | *đo* |
| **E11** | **+ Evolution Reranking** | Tái xếp hạng phân phối xác suất dựa trên 78 họ tiến hóa | 3,590 | *đo* | *đo* | *đo* | *đo* |
| **E12** | **Hierarchical Stage 1** | Phân loại họ tiến hóa (Family Classifier ~78 lớp) | 3,590 | *đo* | *đo* | *đo* | *đo* |
| **E13** | **Resolution 128x128** | Tăng độ phân giải lấy mẫu Nyquist lên 128x128 pixel | dynamic | *đo* | *đo* | *đo* | *đo* |

> [!NOTE]
> Các ô `đo trên Kaggle` sẽ được cập nhật tự động khi chạy file `kaggle/run_experiments.py` hoặc tải `results/experiments.csv` về sau khi train.

---

## 3. Cơ sở Lý thuyết DSP cho các Thành phần Mới

### 3.1. Ma trận đồng xuất mức xám (GLCM - Gray-Level Co-occurrence Matrix)
GLCM thống kê tần suất xuất hiện đồng thời của hai điểm ảnh có mức xám $i$ và $j$ cách nhau khoảng cách $d$ theo góc $\theta$:
$$P(i, j \mid d, \theta) = \frac{1}{N} \sum_{x} \sum_{y} \mathbb{I}[I(x, y) = i \land I(x+dx, y+dy) = j]$$
Từ đó trích xuất 5 chỉ số thống kê bậc hai:
1. **Contrast (Độ tương phản)**: Đo mức độ biến thiên độ sáng cục bộ: $\sum |i-j|^2 P(i,j)$
2. **Dissimilarity (Độ khác biệt)**: Đo độ lệch tuyến tính giữa các pixel lân cận: $\sum |i-j| P(i,j)$
3. **Homogeneity (Độ đồng nhất)**: Phản ánh độ mượt bề mặt: $\sum \frac{P(i,j)}{1 + (i-j)^2}$
4. **Energy / ASM (Độ tập trung năng lượng)**: Phản ánh tính lặp lại kết cấu: $\sum P(i,j)^2$
5. **Correlation (Hệ số tương quan)**: Đo tương quan tuyến tính giữa các điểm xám.

### 3.2. Bộ lọc Gabor (Gabor Filter Bank)
Hàm phản ứng xung của bộ lọc Gabor 2 chiều là tích của hàm bao Gauss và sóng phẳng điều hòa:
$$g(x, y; \lambda, \theta, \psi, \sigma, \gamma) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi\frac{x'}{\lambda} + \psi\right)$$
Trong đó $x' = x\cos\theta + y\sin\theta$, $y' = -x\sin\theta + y\cos\theta$. Phản ứng qua các thang đo tần số ($\lambda$) và góc quay ($\theta$) cho phép thu thập vân kết cấu định hướng độc lập với độ sáng.

### 3.3. Phổ biến đổi Fourier 2 chiều (2D Discrete Fourier Transform)
$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
Phổ biên độ dịch tâm $|F_{\text{shift}}(u, v)|$ được phân tích theo 2 hệ tọa độ cực:
- **Radial Rings ($r_1 \le r < r_2$)**: Đo tổng năng lượng theo các dải tần số (thấp: hình dạng bao quát; cao: biên góc và chi tiết sắc cạnh).
- **Angular Wedges ($\theta_1 \le \theta < \theta_2$)**: Đo phân bố năng lượng theo góc định hướng.

---

## 4. Danh mục Kiểm tra Tính Tái lập (Reproducibility Checklist)

- [x] Tách biệt hoàn toàn bản sao tăng cường (Augmentation) bằng `StratifiedGroupKFold`.
- [x] Không rò rỉ chuẩn hóa (`StandardScaler` chỉ `fit` trên Train fold).
- [x] Cố định hạt giống ngẫu nhiên (`random_state=42`, `torch.manual_seed(42)`).
- [x] Đồng bộ kích thước vector đặc trưng giữa `src/features.py`, `src/predict.py` và `kaggle/notebook_b_train.py`.
- [x] Tích hợp Test-Time Augmentation (Flip TTA) và Evolution Reranking trong suy luận.
- [x] Lưu trữ đầy đủ siêu dữ liệu vào `Model/meta.pkl`, `Model/labels.pkl`, `Model/scaler.pkl`.
