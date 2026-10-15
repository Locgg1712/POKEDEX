# Pokédex DSP Visual – Hệ thống nhận diện Pokémon

##  Giới thiệu

Dự án này xây dựng một hệ thống **nhận diện Pokémon từ ảnh** bằng cách kết hợp:

* Xử lý tín hiệu số (Digital Signal Processing – DSP)
*  Học máy (Machine Learning – ML)

## Cài đặt dự án từ GitHub

### Clone repository

```bash
git clone https://github.com/Locgg1712/POKEDEX.git
cd POKEDEX
```

### Cài thư viện

```bash
pip install -r requirements.txt
```

### Chạy project

```bash
python -m src.main
```
### Chạy giao diện GUI

```bash
python -m src.app
```

Hệ thống nhận đầu vào là một ảnh và trả về:

* Tên Pokémon
* Độ tin cậy (confidence)

Ý tưởng chính là coi ảnh như một **tín hiệu số 2 chiều**, áp dụng DSP để trích xuất đặc trưng, sau đó dùng ML để phân loại.

---

##  Pipeline tổng thể

```
Ảnh đầu vào 
   → Tiền xử lý (DSP)
   → Trích đặc trưng (multi-scale HOG + Color + Fourier + Shape)
   → Chuẩn hóa (Scaling)
   → Phân loại (SVM)
   → Kết quả (Tên + Confidence)
```

---

##  Các thành phần chính

### 1. Tiền xử lý (DSP)

* Resize ảnh về kích thước chuẩn
* Khử nhiễu bằng **Bilateral Filter**
* Tách foreground bằng **Otsu + fallback**
* Phát hiện biên bằng **Canny** (phục vụ feature)

 Mục đích:

* Loại bỏ nhiễu
* Làm nổi bật cấu trúc (biên, hình dạng)

---

### 2. Trích xuất đặc trưng

Hệ thống sử dụng kết hợp các nhóm đặc trưng:

####  HOG (Histogram of Oriented Gradients)

* Mô tả hình dạng và cấu trúc
* Ít bị ảnh hưởng bởi ánh sáng

---

####  HSV Color Histogram

* Mô tả phân bố màu sắc
* Rất hiệu quả với Pokémon có màu đặc trưng (ví dụ: Pikachu màu vàng)

---

####  Fourier Descriptors (DSP)

* Mô tả hình dạng tổng thể dựa trên miền tần số
* Giúp phân biệt các đối tượng theo contour

####  Đặc trưng bổ sung

* HOG ở hai kích thước ô để bắt cả hình dạng tổng thể và chi tiết nhỏ
* Histogram HSV theo từng vùng ảnh
* Hình học contour và Hu Moments để mô tả silhouette
* Lab moments và LBP để bổ sung màu sắc và kết cấu

Trong quá trình huấn luyện, ảnh được tăng cường bằng lật, xoay nhẹ,
zoom vừa phải, thay đổi sáng/màu, nhiễu Gaussian và blur nhẹ. Các bản
tăng cường của cùng ảnh gốc được giữ chung một group khi đánh giá.

---

### 3. Mô hình học máy

* Thuật toán: **SVM (Support Vector Machine)**
* Kernel: RBF
* Tối ưu tham số bằng GridSearchCV

---

##  Kết quả

* Độ chính xác: **~94%**
* Nhận diện tốt các Pokémon có màu đặc trưng
* Nhầm lẫn nhẹ giữa các Pokémon có màu/hình dạng tương tự

---

##  Cấu trúc project

```
POKEDEX/
│── Danh_gia/           #file danh gia (khong bao gom ảnh)
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   └── app.py
│── test/               # ảnh test
├── data/              # (không bao gồm)
├── Model              # (không bao gồm)
├── README.md
├── requirenments.txt
└── .gitignore
```

---

##  Lưu ý

* Dataset không được đưa lên GitHub do dung lượng lớn
* Model không được cung cấp (có thể train lại)
* Dự án tập trung vào sự kết hợp giữa DSP và ML
* Đánh giá hiện không có ảnh để chạy đánh giá
---

## Tải data set để chạy lại model 
```bash
 https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000 
```

## Huấn luyện trên Kaggle

Dataset không nằm trong repository. Có thể huấn luyện lại theo quy trình sau:

1. Tạo một Kaggle Notebook, bật **GPU T4** (hoặc CPU nếu chỉ muốn kiểm tra).
2. Vào **Add Input** và thêm dataset `noodulz/pokemon-dataset-1000`.
3. Chạy các cell sau trong notebook:

```python
!git clone https://github.com/Locgg1712/POKEDEX.git /kaggle/working/POKEDEX
%cd /kaggle/working/POKEDEX
!pip install -q -r requirements.txt
```

Xác định thư mục chứa các thư mục lớp Pokémon:

```python
from pathlib import Path

data_candidates = list(Path("/kaggle/input").glob("**/pokemon-dataset-1000"))
print(data_candidates)
!find /kaggle/input -maxdepth 3 -type d | head -30
```

Chọn đường dẫn có các thư mục như `Bulbasaur`, `Pikachu`, ... rồi chạy Notebook A.
Ví dụ, nếu dataset nằm tại `/kaggle/input/pokemon-dataset-1000`, chạy:

```python
import os
os.environ["POKEDEX_ROOT"] = "/kaggle/working/POKEDEX"
os.environ["POKEDEX_DATA"] = "/kaggle/input/pokemon-dataset-1000"
os.environ["POKEDEX_FEATURES"] = "/kaggle/working/POKEDEX/features.npz"

%run kaggle/notebook_a_features.py
```

Sau khi trích xuất đặc trưng xong, chạy huấn luyện và ablation:

```python
os.environ["POKEDEX_MODEL"] = "/kaggle/working/POKEDEX/Model"
os.environ["POKEDEX_RESULTS"] = "/kaggle/working/POKEDEX/results"

%run kaggle/notebook_b_train.py
```

Kết quả nằm trong `Model/` và `results/`. Để tải về Kaggle, mở panel **Output**
và chọn **Save Version**, hoặc nén kết quả:

```python
!zip -r /kaggle/working/pokedex-results.zip Model results
```

Notebook B chạy 10 cấu hình ablation và ensemble 5 seed nên có thể mất khá lâu.
Nếu chỉ cần mô hình cuối cùng, có thể giữ lại cấu hình `E10` và bỏ qua các cấu hình
trước đó sau khi đã kiểm tra pipeline.

##  Đóng góp chính

* Kết hợp DSP và ML trong bài toán nhận diện ảnh
* Sử dụng Fourier Descriptor (DSP) để mô tả hình dạng
* Pipeline hiệu quả, đạt độ chính xác cao (~94%)

---

##  Công nghệ sử dụng

* Python
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

---

##  Tác giả

* LOCGG1712 – Đồ án Xử lý tín hiệu số

---

##  Kết luận

Dự án cho thấy việc áp dụng **Xử lý tín hiệu số (DSP)** giúp cải thiện đáng kể hiệu quả của mô hình học máy trong bài toán nhận diện ảnh, bằng cách trích xuất các đặc trưng quan trọng và giảm nhiễu dữ liệu đầu vào.

---
