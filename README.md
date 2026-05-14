# Pokédex DSP Visual – Hệ thống nhận diện Pokémon

##  Giới thiệu

Dự án này xây dựng một hệ thống **nhận diện Pokémon từ ảnh** bằng cách kết hợp:

* Xử lý tín hiệu số (Digital Signal Processing – DSP)
*  Học máy (Machine Learning – ML)

## Cài đặt dự án từ GitHub
# Clone repository
git clone https://github.com/Locgg1712/POKEDEX.git
# Cài thư viện
pip install -r requirements.txt

# Chạy project
python main.py

Hệ thống nhận đầu vào là một ảnh và trả về:

* Tên Pokémon
* Độ tin cậy (confidence)

Ý tưởng chính là coi ảnh như một **tín hiệu số 2 chiều**, áp dụng DSP để trích xuất đặc trưng, sau đó dùng ML để phân loại.

---

##  Pipeline tổng thể

```
Ảnh đầu vào 
   → Tiền xử lý (DSP)
   → Trích đặc trưng (HOG + Color + Fourier)
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

Hệ thống sử dụng kết hợp 3 loại đặc trưng:

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
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   └── app.py
│── test/               # ảnh test
├── data/              # (không bao gồm)
├── model.pkl          # (không bao gồm)
├── README.md
└── .gitignore
```

---

##  Lưu ý

* Dataset không được đưa lên GitHub do dung lượng lớn
* Model không được cung cấp (có thể train lại)
* Dự án tập trung vào sự kết hợp giữa DSP và ML

---

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
