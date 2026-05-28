### ĐẠI HỌC QUỐC GIA HÀ NỘI

### TRƯỜNG ĐẠI HỌC CÔNG NGHỆ

### BÁO CÁO DỰ ÁN

# NHẬN DIỆN HÌNH ẢNH POKÉMON

# BẰNG DSP VÀ HỌC MÁY

### Nghiên cứu đối chứng tiền xử lý giữa

### Bộ lọc Cổ điển và Mạng U-Net

```
Thành viên: Cao Văn Lộc — 24020556
Giảng viên: TS. Trần Thị Thúy Quỳnh
```
### Hà Nội, 4/


### Tóm tắt nội dung

Báo cáo trình bày một hệ thống nhận diện hình ảnh Pokémon dựa trên
sự kết hợp giữa **Xử lý tín hiệu số (Digital Signal Processing — DSP)**
và **Học máy (Machine Learning)**. Hệ thống thực hiện phân loại 10 loài
Pokémon thông qua một luồng xử lý tín hiệu chặt chẽ: khử nhiễu bằng
bộ lọc Bilateral, trích xuất biên tự động bằng thuật toán Canny, và tổng
hợp không gian đặc trưng đa chiều bao gồm Fourier Descriptors, HOG
(Histogram of Oriented Gradients) và biểu đồ màu HSV. Mô hình Support
Vector Machine (SVM) với kernel RBF được sử dụng để phân loại, đạt độ
chính xác **95.2%**. Bên cạnh đó, cũng thực hiện một nghiên cứu đối chứng
độc lập nhằm đánh giá khả năng của mạng học sâu U-Net Denoising
Autoencoder và mô hình CNN (đạt độ chính xác lên tới **99.3%** ). Kết quả
thực nghiệm và Ablation Study khẳng định sự kết hợp giữa các đặc trưng
DSP và Bilateral Filter vẫn là lựa chọn vô cùng tối ưu cho hệ thống nhờ khả
năng bảo toàn phân phối màu sắc, tính minh bạch cao và duy trì tốc độ xử
lý tính toán vượt trội so với hộp đen Deep Learning.

**Mã nguồn mở (GitHub):**
▷DSP + SVM:Pokedex_DSP
▷Deep Learning:Pokedex_DeepLearning


## Mục lục

- 1 GIỚI THIỆU
   - 1.1 Đặt vấn đề
   - 1.2 Mục tiêu
   - 1.3 Phạm vi nghiên cứu
- 2 CƠ SỞ LÝ THUYẾT
   - 2.1 Tập dữ liệu Pokémon và tiền xử lý không gian
   - 2.2 Xử lý tín hiệu số trong không gian 2D
      - 2.2.1 Ảnh như một tín hiệu số và nhiễu
      - 2.2.2 Bộ lọc Median Blur
      - 2.2.3 Bộ lọc Bilateral (Lọc phi tuyến giữ biên).
      - 2.2.4 Phát hiện cạnh Canny tự động
   - 2.3 Kỹ thuật chuyển đổi miền tần số: Fourier Descriptors
   - 2.4 Mô hình Học máy và Mạng nơ-ron
      - 2.4.1 Support Vector Machine (SVM)
      - 2.4.2 U-Net Denoising Autoencoder
- 3 PHƯƠNG PHÁP THỰC HIỆN
   - 3.1 Luồng xử lý chính của hệ thống (Pipeline DSP + SVM)
   - 3.2 Kỹ thuật tăng cường dữ liệu (Data Augmentation)
   - 3.3 Thiết lập thực nghiệm đối chứng khử nhiễu (Deep Learning)
- 4 KẾT QUẢ THỰC NGHIỆM
   - 4.1 Hiệu năng tổng thể (Metrics)
   - 4.2 Phân tích per-class và Ma trận nhầm lẫn
   - 4.3 Nghiên cứu bóc tách (Ablation Study)
- 5 THẢO LUẬN
   - 5.1 So sánh đối chứng khử nhiễu: DSP và Deep Learning
   - 5.2 Đánh giá định lượng khử nhiễu: PSNR và SSIM
   - 5.3 Phân tích độ phức tạp tính toán
   - 5.4 Hạn chế của hệ thống
- 6 KẾT LUẬN
   - 6.1 Tóm tắt
   - 6.2 Mã nguồn dự án (Source Code).
   - 6.3 Hướng phát triển tương lai
         -


## 1 GIỚI THIỆU

### 1.1 Đặt vấn đề

Nhận dạng nhân vật hoạt hình trông đơn giản, nhưng không phải vậy. Nhận
dạng các nhân vật hoạt hình như Pokémon khó hơn tưởng tượng – chúng rất
đa dạng về hình dạng, màu sắc và cấu trúc đường viền, và những khác biệt
đó rất quan trọng. Dữ liệu đầu vào thực tế từ webcam hoặc các nguồn trực
tuyến làm tăng thêm độ khó: nhiễu, ánh sáng thay đổi và góc chụp không nhất
quán có thể âm thầm làm giảm độ chính xác nhận dạng nếu quá trình tiền xử
lý không tính đến chúng.

Các pipeline dựa trên mạng nơ-ron (CNN) là lựa chọn hiện đại hiển nhiên,
nhưng chúng tốn nhiều tài nguyên và phần lớn không minh bạch. Việc trích
xuất đặc trưng dựa trên xử lý tín hiệu số (DSP) vẫn giữ vị trí của nó. Thách thức
là thiết kế một pipeline loại bỏ nhiễu mà không làm mất đi các cạnh đặc trưng,
tạo ra các vector đặc trưng ổn định khi xoay, thu phóng và dịch chuyển, và
không bị lỗi trên các ví dụ mà nó không được huấn luyện.

### 1.2 Mục tiêu

Dự án này theo đuổi ba mục tiêu:

- Xây dựng một quy trình nhận dạng Pokémon hoàn chỉnh trên nền tảng xử
    lý tín hiệu số (DSP) kết hợp với học máy cổ điển. Kỹ thuật cốt lõi là xử lý các
    đường viền khép kín như các chuỗi giá trị phức và phân tích chúng bằng
    phép biến đổi Fourier để tách các thành phần hình dạng-tần số.
- Chạy so sánh có kiểm soát ở giai đoạn tiền xử lý - cụ thể là liệu bộ lọc
    phi tuyến tính cổ điển (Bilateral Filter) hay bộ khử nhiễu học sâu (U-Net
    Autoencoder) có làm sạch tín hiệu đầu vào tốt hơn hay không.
- Đo lường bằng số liệu mức độ đóng góp thực tế của từng nhóm đặc trưng.
    Một nghiên cứu phân tích tác động (Ablation Study) bao gồm ba nhóm mô
    tả: hình dạng, kết cấu và màu sắc.

### 1.3 Phạm vi nghiên cứu

Hệ thống xử lý 10 lớp, một lớp cho mỗi loài Pokémon: Bulbasaur, Charmander,
Eevee, Jigglypuff, Magikarp, Meowth, Pikachu, Psyduck, Snorlax và Squirtle.
Hình ảnh có dạng RGB tĩnh với độ phân giải hỗn hợp - một số trên nền trong
suốt sạch sẽ, một số khác trên nền lộn xộn. Toàn bộ quy trình, từ tiền xử lý đến
suy luận, chạy trên CPU tiêu chuẩn mà không cần GPU.

## 2 CƠ SỞ LÝ THUYẾT

### 2.1 Tập dữ liệu Pokémon và tiền xử lý không gian

Bộ dữ liệu có 10 lớp Pokémon, mỗi lớp trong thư mục riêng của nó [ 8 ]. Kích
thước hình ảnh và tỷ lệ khung hình khác nhau đáng kể, vì vậy mọi thứ được
thay đổi kích thước thành 64 × 64 pixel trước khi đi vào đường ống DSP - độ


phân giải không gian đồng nhất mà các bước miền tần số phụ thuộc vào. Bộ
sưu tập thô bắt đầu từNrawhình ảnh. Áp dụng tăng cường dữ liệu ở 11 biến
thể trên mỗi hình ảnh mang lại tổng số đào tạo cho:

```
Ntotal= 11×Nraw (1)
```
Bảng 1 dưới đây chia nhỏ số lượng mẫu trên mỗi lớp trước và sau khi 1
Augmentation:

**Bảng 1.** Phân bố dữ liệu Pokémon 10 lớp trước và sau quá trình Data Augmentation
(× 11 biến thể)

```
Loài Pokémon Ảnh gốc Sau
Augment
```
```
Train
(80%)
```
```
Val (20%)
```
```
Bulbasaur 150 1,650 1,320 330
Charmander 148 1,628 1,302 326
Eevee 142 1,562 1,249 313
Jigglypuff 145 1,595 1,276 319
Magikarp 138 1,518 1,214 304
Meowth 130 1,430 1,144 286
Pikachu 175 1,925 1,540 385
Psyduck 140 1,540 1,232 308
Snorlax 165 1,815 1,452 363
Squirtle 155 1,705 1,364 341
```
```
TỔNG 1,488 16,368 13,094 3,
```
Tập dữ liệu mở rộng, được chia 80/20 với random seed = 42, cho mô hình đủ
biến thể hình học và ánh sáng để SVM có thể tìm thấy ranh giới quyết định rõ
ràng hơn và giữ tốt hơn dữ liệu mà nó chưa thấy.

### 2.2 Xử lý tín hiệu số trong không gian 2D

#### 2.2.1 Ảnh như một tín hiệu số và nhiễu

Hình ảnh thang độ xám từ chuyển đổi RGB hoạt động như một tín hiệu không

gian 2DI(x, y). Ảnh chụp trong thế giới thực hiếm khi xuất hiện rõ ràng - hai
nguồn nhiễu có xu hướng xuất hiện: hiện vật muối và hạt tiêu từ lỗi cảm biến
và nhiễu Gaussian từ ánh sáng mờ hoặc phần cứng CCD/CMOS hoạt động kém.

#### 2.2.2 Bộ lọc Median Blur

Trước khi Bộ lọc Bilateral tham gia, tín hiệu sẽ đi qua Median Blur để xử lý nhiễu
dựa trên xung - muối và hạt tiêu là tác nhân chính. Trong khi một bộ lọc tiêu
chuẩn tính trung bình cường độ pixel lân cận, Median Blur thay vào đó chọn


giá trị trung bình trong một cửa sổ cục bộ. Nó bỏ qua hoàn toàn các ngoại lệ,
đó là lý do tại sao nó xóa nhiễu cô lập mà không kéo các cạnh với nó.

Đối với vùng lân cận quanh pixel đang xétW, pixel đầu ra là :

```
I′(x, y) =median{I(i, j)|(i, j)∈W}
```
Một hạt nhân 3 × 3 chạy ở đây trước Bộ lọc Bilateral. Mục tiêu là bắt nhiễu xung
tần số cao trước khi nó đạt đến giai đoạn phát hiện cạnh và chiết xuất đường
viền - nhiễu mà các bộ lọc dựa trên tính trung bình sẽ lan truyền thay vì loại bỏ.
Lọc Median giữ nguyên các chuyển đổi cường độ sắc nét, điều này quan trọng
khi phân tích Fourier ở hạ lưu phụ thuộc vào ranh giới đường viền chính xác.

#### 2.2.3 Bộ lọc Bilateral (Lọc phi tuyến giữ biên).

Để xử lý nhiễu Gaussian mà không làm mất chi tiết cạnh - các đường viền xác
định từng Pokémon - đường ống sử dụng Bộ lọc Bilateral thay vì bộ lọc Gaussian
tuyến tính tiêu chuẩn [ 1 ]. Cường độ được lọc tại pixelplà:

```
I′(p) =
```
##### 1

```
Wp
```
##### ∑

```
q
```
```
Gs(∥p−q∥)·Gr(|I(p)−I(q)|)·I(q) (2)
```
Trong đóGslà trọng số không gian — giảm dần khiqra xap— cònGrlà trọng
số cường độ sáng, giảm khi chênh lệch độ sáng giữapvàqtăng lên.Wplà hệ
số chuẩn hóa. Chính việc nhân hai Gaussian này lại với nhau tạo ra hành vi
hữu ích của bộ lọc: các vùng màu đồng nhất được làm mịn, còn các viền — nơi
|I(p)−I(q)|tăng đột biến — vẫn giữ nguyên độ sắc nét.

#### 2.2.4 Phát hiện cạnh Canny tự động

Canny truyền thống đòi hỏi phải chỉnh hai ngưỡng bằng tay [ 2 ]. Để tránh việc
phải tinh chỉnh lại mỗi khi điều kiện ánh sáng thay đổi, pipeline áp dụng cơ chế
_Auto-Canny_ — tự động nội suy ngưỡng dựa trên giá trị trung vị của phổ tín hiệu
ảnh:

lower=max(0,(1−σ)·median), upper=min(255,(1 +σ)·median), σ= 0. 33
(3)

Sau đó, phép toán hình thái học _Close_ với kernel 2 × 2 được chạy qua để hàn
gắn các nét đứt và đóng kín những contour còn hở do nhiễu để lại.

### 2.3 Kỹ thuật chuyển đổi miền tần số: Fourier Descriptors

Khi đã có đường viền khép kín, đối tượng được mã hóa thành một tín hiệu số
phức một chiều chạy dọc theo chu vi:

```
z(n) =x(n) +j·y(n) (4)
```
Để tín hiệu của mọi Pokémon có cùng độ dài, đường viền được _resample_ về
N= 128điểm. DFT sau đó được tính [ 3 ]:


```
X(k) =
```
##### N∑− 1

```
n=
```
```
z(n)e−j^2 πkn/N, k= 0, 1 ,... , N− 1 (5)
```
**Phân tích ý nghĩa phổ tần số của Fourier Descriptors:**

DFT chuyển tín hiệu đường viền từ miền không gian sang miền tần số [ 3 ].
Mỗi hệ sốX(k)mang một ý nghĩa hình học cụ thể trong bối cảnh nhận diện
Pokémon:

- **Tần số thấp** (knhỏ) phản ánh hình dạng tổng thể — độ tròn của Jigglypuff,
    hay thân hình bầu bĩnh của Snorlax.
- **Tần số cao** (klớn) phản ánh chi tiết cục bộ: tai nhọn của Pikachu, cạnh sắc
    của móng vuốt.

Hệ thống chỉ giữ 32 hệ số đầu tiên, tương đương áp một **bộ lọc thông thấp**
trong miền tần số — giữ lại cấu trúc hình dạng cốt lõi và loại bỏ nhiễu biên tần
số cao.

Chia choX(1)giúp triệt tiêu ảnh hưởng của tỷ lệ. Lấy biên độ thay vì giữ pha
giúp triệt tiêu ảnh hưởng của góc xoay. Hai bước này kết hợp tạo ra một vector
đặc trưng không bị ảnh hưởng bởi kích thước hay hướng của đối tượng — đúng
yêu cầu cho bài toán phân loại đa lớp.

Trong trường hợp ảnh có nhiều contour, hệ thống tính **weightedaverage** theo
diện tích và ưu tiên đường viền lớn nhất — thường là phần thân chính của
Pokémon.

### 2.4 Mô hình Học máy và Mạng nơ-ron

#### 2.4.1 Support Vector Machine (SVM)

SVM hoạt động bằng cách tìm siêu phẳng có lề lớn nhất để phân tách các lớp
[ 5 ]. Với không gian đặc trưng kết hợp _HOG + HSV + Fourier_ vượt quá 1100 chiều,
cộng thêm ranh giới giữa các loài Pokémon vốn phi tuyến và khó tách bằng mặt
phẳng thẳng, kernel _RBF_ ( _Radial Basis Function_ ) là lựa chọn phù hợp hơn kernel
tuyến tính:

```
K( x , x ′) =exp
```
##### (

```
−γ∥ x − x ′∥^2
```
##### )

##### (6)

Cvàγđược tối ưu thông qua _GridSearchCV_ với _3-fold Cross Validation_. Thay vì
chạy _GridSearch_ trên toàn bộ dữ liệu ngay từ đầu, hệ thống trước tiên thu hẹp
không gian tham số trên tập con gồm 2000 mẫu bằng thư viện _scikit-learn_ [ 7 ].
Sau đó, mô hình tốt nhất được huấn luyện lại trên toàn bộ tập train.

#### 2.4.2 U-Net Denoising Autoencoder

U-Net là kiến trúc _Encoder–Decoder_ , nổi bật nhờ các _Skip Connection_ [ 6 ] — cầu
nối truyền thẳng ma trận đặc trưng từ encoder sang decoder, giúp bù lại phần
cấu trúc không gian bị mất tại _Bottleneck_.

Kiến trúc được sử dụng trong đề tài:


Encoder(3→ 16 →32) → Bottleneck(64) → Decoder(64→ 32 →16) → Conv(16→3)

Mạng được huấn luyện để tối thiểu hóa hàm mất mát MSE giữa ảnh nhiễu và
ảnh gốc, sử dụng _Adam Optimizer_.

## 3 PHƯƠNG PHÁP THỰC HIỆN

### 3.1 Luồng xử lý chính của hệ thống (Pipeline DSP + SVM)

Hệ thống chạy tuần tự qua 4 khối chức năng. Sơ đồ tổng thể và chi tiết độ phức
tạp/mã nguồn được trình bày tại **Hình 1** và **Hình 2**.

**Bước 1 — Tiền xử lý:** Ảnh RGB đầu tiên đi qua _Median Blur_ với kernel 3 × 3 để
loại bỏ nhiễu muối tiêu. Sau đó, _Bilateral Filter_ (d = 9,σcolor= 75,σspace= 75)
được áp dụng nhằm làm mịn các vùng màu đồng nhất nhưng vẫn giữ nguyên
các nét viền đặc trưng.

**Bước 2 — Trích xuất viền:** _Auto-Canny_ quét ảnh để tìm các điểm có gradient
cực đại. Tiếp theo, phép toán hình thái học _Morphology Close_ với kernel 2 × 2
được áp dụng lên bản đồ cạnh nhằm vá các lỗ hổng và khép kín những contour
còn hở.

**Bước 3 — Tổng hợp vector đặc trưng đa miền:** Ba nhóm đặc trưng được
khai thác song song gồm:

- _Fourier_ : sử dụng 32 hệ số bất biến theo tỷ lệ, kết hợp _weightedaverage_ theo
    diện tích.
- _HOG_ : gồm 9 hướng gradient [ 4 ], cell 8 × 8 , block 2 × 2.
- _HSV_ : histogram màu với 16 × 8 × 8 bins.

Ba nhóm đặc trưng sau đó được ghép lại thành vector đặc trưng có kích thước
xấp xỉ 1132 chiều.

**Bước4—Chuẩnhóavàhuấnluyện:** Chuẩn hóa _Z-score_ được sử dụng để cân
bằng tỷ lệ giữa các nhóm đặc trưng, tránh việc HOG lấn át Fourier. Sau đó, SVM
được tinh chỉnh bằng _GridSearchCV_ với _3-fold Cross Validation_ để tìm các tham
số tối ưuCvàγ, trước khi huấn luyện lại trên toàn bộ tập train.

```
xscaled=
```
```
x−μ
σ
```

```
Hình 1. Sơ đồ luồng xử lý tổng thể của hệ thống DSP + SVM.
```
**Liên kết giữa pipeline và hiện thực mã nguồn:**

Hệ thống được xây dựng theo kiến trúc module hóa — mỗi bước xử lý tương
ứng với một thành phần mã nguồn riêng, kèm độ phức tạp thuật toán xác định.
Cách tổ chức này giúp từng block DSP có thể được chỉnh sửa, kiểm thử hoặc
tái sử dụng độc lập mà không ảnh hưởng đến phần còn lại. Toàn bộ mã nguồn
được đính kèm tại GitHub của dự án.


```
Hình 2. Bảng phân tích độ phức tạp thuật toán và mapping mã nguồn.
```
### 3.2 Kỹ thuật tăng cường dữ liệu (Data Augmentation)

Dữ liệu đồ họa thường không có nhiều mẫu — để bù lại, hệ thống áp dụng
augmentation offline, tạo đủ biến thể để SVM không học vẹt trên tập train. Mỗi
ảnh gốc được mở rộng thành 11 biến thể, chi tiết tại Bảng 2.


```
Bảng 2. Các phép biến đổi Augmentation được áp dụng
```
```
STT Loại biến
đổi
```
```
Mô tả kỹ thuật Mục đích
```
```
1 Gốc Không thay đổi Mẫu tham chiếu chuẩn
2 Hình học Lật ngang (Flip) Bất biến đối xứng
trái-phải
3 Hình học Xoay+20° Giả lập sai lệch góc chụp
4 Hình học Xoay− 20 ° Giả lập sai lệch góc chụp
5 Hình học Thu nhỏ 80% Kiểm chứng chuẩn hóa
Fourier
6 Hình học Shear ngang 15% Bóp méo phối cảnh phi
tuyến
7 Hình học Translate(+8,+8)px Đối tượng không nằm ở
tâm
8 Hình học Translate(− 8 ,−8)px Đối tượng không nằm ở
tâm
9 Màu sắc Sáng hơn (α= 1. 3 ,
β= 20)
```
```
Tính chống chịu HSV
Histogram
10 Màu sắc Tối hơn (α= 0. 7 ,
β=− 20 )
```
```
Tính chống chịu HSV
Histogram
11 Nhiễu Gaussian noise (σ= 8) Ép Bilateral + HOG khắc
nghiệt
```
### 3.3 Thiết lập thực nghiệm đối chứng khử nhiễu (Deep Learning)

Để đánh giá _Bilateral Filter_ ở Bước 1 một cách có đối chiếu, một mạng U-Net
được xây dựng song song bằng _PyTorch_. Thay vì sử dụng tập nhiễu cố định, hệ
thống áp dụng cơ chế _On-the-fly Noise_ — mỗi epoch tự động bơm ngẫu nhiên
nhiễu Gaussian ( 10 %– 30 %) và nhiễu muối tiêu ( 2 %– 8 %) vào ảnh sạch, buộc mạng
phải thích nghi với biên độ nhiễu thay đổi liên tục.

U-Net được huấn luyện để tối thiểu hóa hàm mất mát MSE giữa ảnh đầu ra và
ảnh gốc, sử dụng _AdamOptimizer_ kết hợp với _ReduceLROnPlateau_ (patience= 3)
nhằm điều chỉnh _learning rate_.

Checkpoint tốt nhất được lưu dựa trên giá trị val_loss trong suốt 15 epochs.

## 4 KẾT QUẢ THỰC NGHIỆM

### 4.1 Hiệu năng tổng thể (Metrics)

SVM được kiểm tra trên tập _test_ — gồm 20 % dữ liệu mà mô hình chưa từng thấy
trong quá trình huấn luyện. Bảng 3 trình bày kết quả theo bốn chỉ số phân loại.


```
Bảng 3. Hiệu năng tổng thể của pipeline DSP + SVM so với CNN
```
```
Chỉ số đánh giá (Metric) DSP + SVM Deep Learning
(CNN)
```
```
Độ chính xác tổng thể
(Accuracy)
```
##### 95.2% 99.3%

```
Độ chuẩn xác (Precision) 95.5% 99.4%
Độ bao phủ (Recall) 95.1% 99.3%
Điểm F1-Score trung bình
(F1-Macro)
```
##### 95.3% 99.3%

_Precision_ ( 95. 5 %) và _Recall_ ( 95. 1 %) gần như bằng nhau , mô hình không bị kéo
lệch về phía nào, không ưu tiên tránh bỏ sót hay tránh báo nhầm hơn cái kia.
Deep Learning có thể đẩy lên 99. 3 %, nhưng đổi lại cần GPU. Pipeline DSP thấp
hơn 4 điểm phần trăm, chạy hoàn toàn trên CPU.

### 4.2 Phân tích per-class và Ma trận nhầm lẫn

Ma trận nhầm lẫn cho thấy lỗi phân loại không phân bố đều. Các lớp có ngoại
hình đặc trưng rõ — thân hình cồng kềnh của Snorlax, màu vàng nổi bật của
Pikachu, hay mai rùa của Squirtle — gần như không bị nhầm, lần lượt đạt 190,
189 và 182 mẫu đúng.

```
Bảng 4. Kết quả phân loại per-class của mô hình SVM
```
```
Loài
Pokémon
```
```
Đúng /
Tổng
```
```
Acc (%) Nhầm chính sang
```
```
Bulbasaur 147 / 158 93.0% Squirtle (2)
Charmander 158 / 165 95.8% Eevee (2), Pikachu (3)
Eevee 129 / 136 94.9% Snorlax (5)
Jigglypuff 157 / 165 95.2% Snorlax (7)
Magikarp 145 / 154 94.2% Bulbasaur (2), Snorlax (5)
Meowth 102 / 117 87.2% Jigglypuff (2), Magikarp (3),
Snorlax (8)
Pikachu 189 / 194 97.4% Eevee (1), Snorlax (2)
Psyduck 144 / 150 96.0% Bulbasaur (2), Pikachu (2)
Snorlax 190 / 194 97.9% Eevee (1), Magikarp (1)
Squirtle 182 / 185 98.4% Psyduck (1), Pikachu (2)
```

**Hình 3.** Ma trận nhầm lẫn (Confusion Matrix) của mô hình phân loại đa lớp SVM —
Accuracy: 95.2%.

**Phân tích Hình 3 – Ma trận nhầm lẫn:**

Nhìn vào ma trận nhầm lẫn (Hình 3 ), một số quy luật hiện ra khá rõ:

- **Đường chéo hội tụ:** Giá trị trên đường chéo chính cao đồng đều — mô
    hình phân loại đúng phần lớn mẫu trong tập validation, không có lớp nào
    bị bỏ rơi hoàn toàn.
- **Vùng dễ nhầm:** Meowth là điểm yếu rõ nhất, chỉ đạt 87. 2 %. Lớp này hay
    bị nhầm sang Jigglypuff hoặc Snorlax — không phải ngẫu nhiên, vì cả ba
    đều có cấu trúc cơ thể tròn, cong. Hệ quả là các hệ số Fourier tần số thấp
    của Meowth bị giao thoa với hai lớp kia. Màu lông nhạt càng làm vấn đề
    trầm trọng hơn — _HSVHistogram_ không tạo được đỉnh đặc trưng đủ mạnh
    để phân biệt.
- **Lớp phân tách sắc nét:** Lớp phân tách sắc nét: Pikachu (vàng tươi) và
    Squirtle (xanh dương, mai rùa) gần như không bị nhầm. Màu sắc và hình
    dạng của hai lớp này quá riêng biệt — HSV và Fourier cùng lúc đẩy chúng
    ra xa phần còn lại của không gian đặc trưng.

Kết quả này cho thấy việc kết hợp ba miền đặc trưng — tần số, không gian và
phân phối màu — giúp mô hình xây dựng được ranh giới phân lớp thực sự có
ý nghĩa, không chỉ khớp dữ liệu train.


### 4.3 Nghiên cứu bóc tách (Ablation Study)

Để kiểm tra xem từng nhóm đặc trưng thực sự đóng góp bao nhiêu, một
Ablation Study được tiến hành — lần lượt thêm và bớt HOG, HSV, Fourier khỏi
pipeline rồi đo lại độ chính xác của SVM. Kết quả được tổng hợp tại Bảng 5.

**Bảng 5.** Ablation Study — Tác động và đóng góp của từng nhóm đặc trưng DSP*so
với cấu hình chỉ HOG

```
Cấu hình Đặc trưng Kích thước
Vector
```
```
Accuracy ∆ so với
trước
```
```
Chỉ HOG (Không gian) ≈1764 chiều 84.1% —
HOG + HSV Histogram (thêm
Màu sắc)
```
```
≈2788 chiều 91.5% +7. 4 %
```
```
HOG + Fourier (Không gian +
Tần số)
```
```
≈1796 chiều 89.2% +5. 1 %*
```
```
HOG + HSV + Fourier
(Pipeline đầy đủ)
```
##### ≈ 2820

```
chiều
```
##### 95.2% +3. 7 %

Chỉ dùng _HOG_ , mô hình dừng ở 84. 1 % — cấu trúc viền cục bộ một mình không
đủ để phân biệt các loài có hình dạng tương đồng.

Thêm _HSV_ tạo ra bước nhảy lớn nhất:+7. 4 %, không bất ngờ, vì màu sắc là thứ
phân biệt Pokémon nhanh nhất ngay cả với mắt người.

_Fourier Descriptors_ đóng góp thêm+3. 7 % — phần tăng nhỏ hơn nhưng có vai
trò rõ ràng: hình dáng tổng thể giải quyết được những ca mà _HOG_ và _HSV_ cùng
bó tay.

Ba nhóm gộp lại cho kết quả 95. 2 %.

## 5 THẢO LUẬN

### 5.1 So sánh đối chứng khử nhiễu: DSP và Deep Learning

Quan sát Hình 5 từ thực nghiệm đối chứng, ta rút ra các luận điểm quan trọng
về cách hai triết lý xử lý hình ảnh giải quyết bài toán.


```
Bảng 6. So sánh chi tiết Bilateral Filter (DSP) vs. U-Net Denoiser (Deep Learning)
```
**Tiêu chí so sánh Bilateral Filter (DSP) U-Net Denoiser (DL)**

Phục hồi cạnh nhỏ
(Micro-edges)

```
Trung bình — làm mờ chi
tiết rất nhỏ
```
```
Vượt trội — tái tạo sắc
nét tai nhọn, mắt
Pikachu
```
Độ ổn định màu
sắc (Color Stability)

```
Xuất sắc — giữ 100% mã
màu HEX gốc
```
```
Yếu — có hiện tượng
color shift (Bulbasaur)
```
Chi phí tính toán Vài ms trên CPU
(convolution ma trận)

```
Forward pass qua hàng
chục nghìn tham số
```
Phù hợp với
pipeline HSV +
Fourier

```
Tối ưu — màu chính xác,
biên tần số thấp đủ
```
```
Rủi ro — color shift phá
hủy HSV vector
```
RAM / VRAM Cực thấp — chạy trên
CPU thông thường

```
Yêu cầu RAM/VRAM cao
hơn hàng trăm lần
```
```
(a) Bulbasaur + Gaussian noise (b) Pikachu + Gaussian noise
```
```
Hình 4. Ảnh nhiễu Gaussian đầu vào.
```

```
(a) Bulbasaur — Khử nhiễu DSP (Bilateral) (b) Pikachu — Khử nhiễu DSP (Bilateral)
```
```
(c) Bulbasaur — Khử nhiễu DL (U-Net) (d) Pikachu — Khử nhiễu DL (U-Net)
```
**Hình 5.** Kết quả trực quan thực nghiệm đối chứng khử nhiễu trên mẫu Bulbasaur và
Pikachu.

U-Net với Skip Connection tái tạo chi tiết nhỏ tốt hơn rõ rệt — viền tai và ranh
giới mắt của Pikachu sắc nét hơn hẳn so với Bilateral Filter. Nhưng đây cũng là
chỗ mạng nhỏ này tự làm hỏng mình: trên ảnh Bulbasaur, U-Net sinh ra hiện
tượng _color shift_ khiến màu xanh ngọc gốc bị trầm đi đáng kể. Với một pipeline
dựa vào HSV Histogram làm đặc trưng lõi, sai lệch màu không phải lỗi nhỏ —
nó phá vector nhận dạng ngay từ đầu vào. Bilateral Filter không tái tạo cạnh
sắc bằng, nhưng hoạt động thuần túy trên trung bình có trọng số của pixel lân
cận nên mã màu gốc giữ nguyên hoàn toàn.


**Kếtluận** , Pipeline này xoay quanh _HSV_ và 32 hệ số Fourier tần số thấp, nên giữ
màu quan trọng hơn phục hồi từng nét viền nhỏ. _Bilateral Filter_ phù hợp ở đây
vì nó không làm lệch màu đầu vào — đơn giản vậy thôi. Nếu thay SVM bằng
CNN end-to-end, bài toán sẽ khác: CNN tự học đặc trưng nên không phụ thuộc
vào độ ổn định màu HSV, lúc đó color shift của U-Net không còn là vấn đề.

### 5.2 Đánh giá định lượng khử nhiễu: PSNR và SSIM

Hai chỉ số sau được sử dụng để so sánh chất lượng khử nhiễu giữa hai phương
pháp:

- **PSNR** ( _Peak Signal-to-Noise Ratio_ ): Tỷ lệ tín hiệu đỉnh trên nhiễu, đơn vị dB.
    Giá trị càng cao thì ảnh khôi phục càng gần với ảnh gốc ở mức từng pixel.
- **SSIM** ( _Structural Similarity Index_ ): Đo mức độ tương đồng về cấu trúc, độ
    sáng và độ tương phản giữa hai ảnh. Chỉ số nằm trong khoảng từ 0 đến
    1 , với 1 biểu thị mức khớp hoàn toàn.

```
Bảng 7. Kết quả PSNR và SSIM theo từng loại nhiễu
```
```
Loài nhiễu PSNR DSP
(dB)
```
```
SSIM DSP PSNR U-Net
(dB)
```
```
SSIM U-Net
```
```
Gaussian Noise 20.06 0.4477 19.68 0.
Salt & Pepper Noise 18.13 0.5573 18.06 0.
Speckle Noise 19.07 0.4371 19.06 0.
```
```
Bảng 8. Tổng hợp trung bình toàn bộ tập thử nghiệm
```
```
Phương pháp PSNR trung bình
(dB)
```
```
SSIM trung bình
```
```
DSP (Bilateral Filter) 19.09 0.
U-Net (Deep Learning) 18.93 0.
```
```
Nhận xét và phân tích:
```
- **PSNR:** _Bilateral Filter_ Bilateral Filter nhỉnh hơn nhẹ — 19. 09 dB so với 18. 93
    dB của U-Net. Hợp lý, vì Bilateral hoạt động trực tiếp trên không gian pixel
    nên kiểm soát sai số tuyệt đối tốt hơn. U-Net nhỏ (bottleneck 64 channels)
    đôi khi tái tạo pixel theo phân phối trung bình học được, dẫn đến lệch pixel
    — color shift trên Bulbasaur là ví dụ điển hình.
- **SSIM:** U-Net thắng rõ — 0.5045 so với 0.4807. SSIM đo chất lượng bảo
    toàn cấu trúc và đường biên, không chỉ sai số pixel, nên kết quả này khớp
    với quan sát định tính: Skip Connection giúp U-Net tái tạo cạnh nhỏ sắc
    nét hơn, dù kèm theo lệch màu.
- **Theo từng loại nhiễu:**


- Với _Gaussian Noise_ , khoảng cách giữa hai phương pháp thể hiện rõ
    nhất về SSIM ( 0. 4477 so với 0. 4807 ). U-Net có lợi thế vì được huấn luyện
    trực tiếp trên phân phối nhiễu này.
- Với nhiễu _Salt & Pepper_ , chênh lệch thu hẹp đáng kể do _Median Blur_ ở
    đầu pipeline đã loại bỏ phần lớn nhiễu trước khi Bilateral Filter được
    áp dụng.
- Với _Speckle Noise_ , hai phương pháp cho hiệu quả gần như tương
    đương.

**Tómlại** , PSNR và SSIM phản ánh hai tiêu chí thiết kế khác nhau — pixel fidelity
và structural similarity. Pipeline hiện tại dùng HSV Histogram làm đặc trưng lõi,
nên giữ màu quan trọng hơn giữ cạnh. Đó là lý do Bilateral Filter phù hợp hơn
ở đây, và cũng giải thích tại sao DSP + SVM vẫn đạt 95.2% với bộ lọc đơn giản
hơn về kiến trúc.

### 5.3 Phân tích độ phức tạp tính toán

Pipeline _DSP + SVM_ gọn nhẹ không chỉ nhờ kiến trúc đơn giản, mà còn vì từng
thành phần đều có độ phức tạp được kiểm soát rõ ràng. Chi tiết được trình bày
tại Hình 2 :

- **FFT:** Độ phức tạpO(NlogN)vớiN= 128điểm contour. Biến đổi tần số có
    thể thực thi nhanh và ổn định ngay cả trên CPU đơn lõi.
- **HOG:** Độ phức tạpO(H×W), tuyến tính theo số lượng pixel của ảnh. Việc
    resize ảnh về kích thước 64 × 64 giúp chi phí tính toán luôn ở mức thấp.
- **SVM với kernel RBF:**
    - _Huấn luyện:_ Độ phức tạp từO(n^2 )đếnO(n^3 )tùy theo tham sốC. Đây
       cũng là lý do _GridSearch_ được thực hiện trước trên tập con gồm 2000
       mẫu.
    - _Inference:_ Độ phức tạpO(nSV), chỉ phụ thuộc vào số lượng _Support_
       _Vectors_ mà mô hình giữ lại để xác định ranh giới phân lớp.

Trên một lõi CPU thông thường, toàn bộ pipeline từ load ảnh đến kết quả phân
loại mất vài mili-giây. Không cần GPU, không cần VRAM — thực tế đủ nhẹ để
chạy nhúng trong ứng dụng thời gian thực quy mô nhỏ mà không cần điều
chỉnh thêm.

### 5.4 Hạn chế của hệ thống

Pipeline hiện tại vẫn tồn tại một số giới hạn thực tế:

- **Đa vật thể:** Khi hai Pokémon xuất hiện gần nhau, _Canny_ dễ tạo ra các
    contour nhập nhằng hoặc chồng lẫn lên nhau. Điều này khiến _Fourier_
    _Descriptors_ thu được tín hiệu không còn đại diện rõ ràng cho bất kỳ loài
    nào.
- **Nềnphứctạp:** Ảnh có texture dày đặc dễ đánh lừa Canny bắt đường viền
    nền thay vì Pokémon, kéo Fourier Descriptors lệch theo.


- **SVM khó mở rộng:** Chi phí suy luận tăng theoO(n×nSV). Mô hình hoạt
    động hiệu quả với khoảng 10 lớp, nhưng sẽ khó đáp ứng yêu cầu thời gian
    thực nếu mở rộng lên hàng trăm loài Pokémon.
- **Số lớp cố định:** Khi thêm loài mới, hệ thống phải huấn luyện lại toàn bộ
    mô hình từ đầu. Pipeline hiện chưa hỗ trợ cơ chế _incremental learning_.

## 6 KẾT LUẬN

### 6.1 Tóm tắt

Hệ thống đạt độ chính xác 95. 2 % trên 10 lớp Pokémon, đúng với mục tiêu đề ra.
Nhưng con số không phải điểm đáng chú ý nhất.

**BilateralFilter** giữ màu nguyên vẹn, **FourierDescriptors** bóc tách hình dạng
tổng thể, **HOG** nắm chi tiết cục bộ — mỗi thành phần có vai trò xác định, và khi
ghép lại, pipeline hoạt động đúng như thiết kế, không phải nhờ may mắn.

**Ablation Study** và đối chiếu với U-Net đều dẫn về cùng một nhận xét: **Deep
Learning** đạt 99. 3 % cao hơn, và con số đó là thật.

Nhưng nó cần GPU, và không cho biết nó quyết định dựa vào gì. Với bài toán 10
lớp chạy trên phần cứng tiêu chuẩn và cần khả năng diễn giải, 4 điểm phần trăm
đó không đủ để đánh đổi. Meowth đạt thấp nhất ( 87. 2 %) không phải vì pipeline
yếu — màu lông nhạt và thân tròn của nó trùng đặc trưng với Jigglypuff và
Snorlax, đây là vấn đề dữ liệu, không phải mô hình.

```
Pipeline DSP + SVM — Kết quả tổng kết
```
```
Accuracy: 95.2% Precision: 95.5% Recall: 95.1% F1-Macro: 95.3%
Lớp tốt nhất: Squirtle 98.4% | Lớp khó nhất: Meowth 87.2%
So sánh Deep Learning (CNN): Đạt 99.3%
Augmentation 11 × | Đặc trưng≈1132 chiều | SVM kernel RBF
```
### 6.2 Mã nguồn dự án (Source Code).

Mã nguồn của cả hai hướng tiếp cận đều công khai trên GitHub:

- **Hệ thống Pokédex DSP + SVM:** Pokedex_DSP
- **Hệ thống Pokédex Deep Learning:** Pokedex_DeepLearning

### 6.3 Hướng phát triển tương lai

- **Xửlýđavậtthể:** Tích hợp Sliding Window + NMS để quét toàn khung hình,
    hoặc Watershed để phân tách vật thể trước khi đưa vào luồng Fourier — cả
    hai đều giải quyết được trường hợp nhiều Pokémon xuất hiện cùng lúc.
- **Mở rộng lên CNN end-to-end:** Nếu tài nguyên phần cứng cho phép, thay
    thế SVM bằng MobileNetV2 hoặc EfficientNet, kết hợp với U-Net Denoiser
    đã có. Trong bối cảnh này, color shift của U-Net sẽ được CNN học cách bù


```
đắp tự động. Áp dụng Focal Loss nếu cần xử lý mất cân bằng lớp.
```
- **Edge AI (IoT):** Nén trọng số SVM rồi nhúng toàn bộ lên Raspberry Pi hoặc
    ESP32-S3 — bước gần nhất đến một Pokédex phần cứng thực sự.
- **Few-ShotLearning:** Prototypical Networks cho phép nhận diện loài mới chỉ
    từ vài ảnh, không đụng đến phần còn lại của pipeline.
- **Tích hợp Grad-CAM:** Thêm lớp trực quan hóa để biết mô hình đang nhìn
    vào đâu khi đưa ra dự đoán — hữu ích cả để debug lẫn để giải thích kết quả
    với người dùng cuối.
- **Tổng quát hóa sang domain khác (Domain Generalization):** Pipeline
    được xây và kiểm chứng trên đồ họa hoạt hình. Câu hỏi mở là liệu Fourier
    + HSV có giữ được hiệu năng khi chuyển sang ký tự viết tay, biển báo giao
    thông hay logo thương hiệu — những domain có cấu trúc đường viền và
    phân phối màu không quá xa, nhưng chưa được kiểm chứng.

## Tài liệu

```
[1] Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color
images. ProceedingsoftheIEEEInternationalConferenceonComputerVision ,
839–846.
```
```
[2] Canny, J. (1986). A computational approach to edge detection. IEEE
Transactions on Pattern Analysis and Machine Intelligence , (6), 679–698.
```
```
[3] Zahn, C. T., & Roskies, R. Z. (1972). Fourier descriptors for plane closed
curves. IEEE Transactions on Computers , 100(3), 269–281.
```
```
[4] Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human
detection. IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) , 886–893.
```
```
[5] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning ,
20(3), 273–297.
```
```
[6] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional
networks for biomedical image segmentation. Medical Image Computing
and Computer-Assisted Intervention (MICCAI) , 234–241.
```
```
[7] Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal
of Machine Learning Research , 12, 2825–2830.
```
```
[8] Saxena, R. R., Nieters, E., & Mamudu, I. (2025). Pokémondium: A machine
learning approach to detecting images of Pokémon. TechRxiv.
```

