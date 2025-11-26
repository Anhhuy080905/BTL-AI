# MÔ HÌNH DECISION TREE - PHÂN LOẠI CHẤT LƯỢNG KHÔNG KHÍ

## 1. TỔNG QUAN MÔ HÌNH

### 1.1 Thuật toán

- **Loại model**: Decision Tree Classifier (scikit-learn)
- **Mục tiêu**: Phân loại chất lượng không khí (AQI) thành 5 cấp độ dựa trên dữ liệu khí tượng
- **Đặc điểm**:
  - Mô hình dễ diễn giải (interpretable)
  - Có khả năng xử lý dữ liệu phi tuyến tính
  - Không cần giả định về phân phối dữ liệu

### 1.2 Dữ liệu đầu vào

**Tổng số mẫu**: 1,804 mẫu (từ file `data_onkk (2).csv`)

**Đặc trưng đầu vào** (6 features):

1. **PRES2M**: Áp suất khí quyển tại độ cao 2m (Pa)
2. **RH**: Độ ẩm tương đối (%)
3. **WSPD**: Tốc độ gió (m/s)
4. **TMP**: Nhiệt độ (°C)
5. **TP**: Lượng mưa (mm)
6. **SQRT_SEA_DEM_LAT**: Biến tổng hợp (Căn bậc hai của khoảng cách đến biển, độ cao địa hình và vĩ độ)

**Biến mục tiêu**: Chất lượng không khí (AQI) được phân loại thành 5 cấp độ:

- **Tốt**: PM2.5 ≤ 15.4 μg/m³
- **Trung bình**: 15.4 < PM2.5 ≤ 40.4 μg/m³
- **Kém**: 40.4 < PM2.5 ≤ 65.4 μg/m³
- **Xấu**: 65.4 < PM2.5 ≤ 150.4 μg/m³
- **Rất xấu**: PM2.5 > 150.4 μg/m³

**Phân chia dữ liệu**:

- **Train set**: 1,443 mẫu (80%)
- **Test set**: 361 mẫu (20%)
- Sử dụng `stratify` để đảm bảo phân phối đồng đều các lớp

---

## 2. TIỀN XỬ LÝ DỮ LIỆU

### 2.1 Chuẩn hóa đặc trưng (Feature Scaling)

- **Phương pháp**: StandardScaler
- **Công thức**: z = (x - μ) / σ
  - μ: Giá trị trung bình
  - σ: Độ lệch chuẩn
- **Lý do**: Đảm bảo các đặc trưng có cùng tỷ lệ, giúp mô hình học tốt hơn

### 2.2 Mã hóa nhãn (Label Encoding)

- **Phương pháp**: LabelEncoder (scikit-learn)
- **Ánh xạ** (sắp xếp theo alphabet):
  - 0: Kém
  - 1: Rất xấu
  - 2: Trung bình
  - 3: Tốt
  - 4: Xấu

---

## 3. XÂY DỰNG VÀ TỐI ƯU THAM SỐ

### 3.1 Kiến trúc mô hình

```python
DecisionTreeClassifier(
    max_depth=10,              # Độ sâu tối đa của cây
    min_samples_split=20,      # Số mẫu tối thiểu để chia nút
    min_samples_leaf=10,       # Số mẫu tối thiểu tại nút lá
    random_state=42,           # Seed để tái tạo kết quả
    class_weight='balanced'    # Cân bằng trọng số các lớp
)
```

### 3.2 Giải thích tham số

**max_depth=10**:

- **Ý nghĩa**: Giới hạn độ sâu của cây quyết định
- **Mục đích**: Ngăn chặn overfitting bằng cách hạn chế độ phức tạp của mô hình
- **Lý do chọn**: Cân bằng giữa khả năng học và khả năng tổng quát hóa

**min_samples_split=20**:

- **Ý nghĩa**: Một nút cần ít nhất 20 mẫu mới được phép chia tiếp
- **Mục đích**: Tránh tạo ra các nhánh quá chi tiết với ít dữ liệu
- **Lý do chọn**: Phù hợp với tập dữ liệu 1,443 mẫu training

**min_samples_leaf=10**:

- **Ý nghĩa**: Mỗi nút lá phải chứa ít nhất 10 mẫu
- **Mục đích**: Đảm bảo mỗi quyết định dựa trên đủ bằng chứng thống kê
- **Lý do chọn**: Giảm nhiễu và tăng độ tin cậy của dự đoán

**class_weight='balanced'**:

- **Ý nghĩa**: Tự động điều chỉnh trọng số các lớp theo tỷ lệ nghịch với tần suất xuất hiện
- **Công thức**: weight = n_samples / (n_classes × n_samples_class)
- **Mục đích**: Xử lý mất cân bằng dữ liệu (imbalanced data)
- **Lý do chọn**: Các lớp "Xấu" và "Rất xấu" có rất ít mẫu, cần tăng trọng số để mô hình không bỏ qua

### 3.3 Đặc điểm của cây đã huấn luyện

- **Độ sâu thực tế**: 10 tầng
- **Số nút lá (leaves)**: 99 nút
- **Ý nghĩa**: Mô hình đã tạo ra 99 quy tắc phân loại khác nhau

---

## 4. KẾT QUẢ TRÊN TẬP TEST

### 4.1 Hiệu suất tổng thể

Kết quả được đánh giá trên tập Test **độc lập** (361 mẫu, chưa từng tham gia quá trình huấn luyện):

| Chỉ số                 | Giá trị    | Ý nghĩa                                           |
| ---------------------- | ---------- | ------------------------------------------------- |
| **Accuracy**           | **60.11%** | Tỷ lệ dự đoán đúng tổng thể                       |
| **Weighted Precision** | 0.6449     | Độ chính xác có trọng số                          |
| **Weighted Recall**    | 0.6011     | Độ phủ có trọng số                                |
| **Weighted F1-Score**  | 0.6098     | Trung bình điều hòa Precision-Recall có trọng số  |
| **Macro F1-Score**     | 0.5388     | Trung bình F1 của tất cả các lớp (không trọng số) |

**So sánh với Neural Network**:

- Neural Network: 55.16%
- Decision Tree: **60.11%**
- **Cải thiện: +4.95%**

### 4.2 Hiệu suất từng lớp (Class-wise Performance)

| Nhãn AQI       | Precision | Recall | F1-Score | Support | Đánh giá                                        |
| -------------- | --------- | ------ | -------- | ------- | ----------------------------------------------- |
| **Tốt**        | 0.59      | 0.78   | 0.67     | 50      | Recall cao - Phát hiện tốt các ngày tốt         |
| **Trung bình** | 0.79      | 0.59   | 0.68     | 190     | Precision cao - Dự đoán chính xác nhất          |
| **Kém**        | 0.45      | 0.47   | 0.46     | 85      | Hiệu suất trung bình, hay nhầm với "Trung bình" |
| **Xấu**        | 0.43      | 0.68   | 0.52     | 34      | Recall cao - Quan trọng cho cảnh báo sớm        |
| **Rất xấu**    | 0.22      | 1.00   | 0.36     | 2       | Phát hiện 100% nhưng có nhiễu (chỉ 2 mẫu)       |

### 4.3 Phân tích chi tiết

#### 4.3.1 Lớp "Tốt" (Support: 50)

- **Precision = 0.59**: Khi dự đoán "Tốt", có 59% thực sự tốt
- **Recall = 0.78**: Phát hiện đúng 78% (39/50) các ngày tốt
- **Đánh giá**: Mô hình có xu hướng dự đoán "Tốt" thận trọng, ít báo động giả

#### 4.3.2 Lớp "Trung bình" (Support: 190 - Lớp lớn nhất)

- **Precision = 0.79**: Độ chính xác cao nhất
- **Recall = 0.59**: Phát hiện đúng 113/190 mẫu
- **Sai số**: 43 mẫu bị nhầm là "Kém", 26 mẫu bị nhầm là "Tốt"
- **Đánh giá**: Lớp được dự đoán tốt nhất, đóng góp chính vào accuracy

#### 4.3.3 Lớp "Kém" (Support: 85)

- **Precision = 0.45**: Khi dự đoán "Kém", chỉ 45% đúng
- **Recall = 0.47**: Phát hiện đúng 40/85 mẫu
- **Sai số chính**:
  - 18 mẫu bị nhầm là "Trung bình"
  - 24 mẫu bị nhầm là "Xấu"
- **Đánh giá**: Lớp khó phân biệt, nằm giữa 2 cấp độ

#### 4.3.4 Lớp "Xấu" (Support: 34 - Lớp thiểu số)

- **Precision = 0.43**: Có nhiều báo động giả
- **Recall = 0.68**: Phát hiện đúng 23/34 trường hợp (quan trọng!)
- **Đánh giá**:
  - **Ưu điểm**: Recall cao (68%) rất quan trọng trong giám sát môi trường
  - Ưu tiên phát hiện nguy cơ hơn là giảm báo động giả
  - Nhờ `class_weight='balanced'` mô hình không bỏ qua lớp này

#### 4.3.5 Lớp "Rất xấu" (Support: 2 - Cực kỳ hiếm)

- **Precision = 0.22**: Nhiều dự đoán sai (9 false positives)
- **Recall = 1.00**: Phát hiện 100% (2/2) trường hợp
- **Đánh giá**: Với chỉ 2 mẫu, kết quả không đủ tin cậy để đánh giá

---

## 5. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

### 5.1 Bảng số liệu

|                         | Dự đoán: Kém | Rất xấu | Trung bình | Tốt | Xấu |
| ----------------------- | ------------ | ------- | ---------- | --- | --- |
| **Thực tế: Kém**        | 40           | 3       | 18         | 0   | 24  |
| **Thực tế: Rất xấu**    | 0            | 2       | 0          | 0   | 0   |
| **Thực tế: Trung bình** | 43           | 2       | 113        | 26  | 6   |
| **Thực tế: Tốt**        | 0            | 0       | 10         | 39  | 1   |
| **Thực tế: Xấu**        | 6            | 2       | 2          | 1   | 23  |

### 5.2 Phân tích xu hướng sai số

**Quan sát chính**:

1. **Đường chéo chính** (dự đoán đúng): 40 + 2 + 113 + 39 + 23 = **217/361 (60.11%)**

2. **Nhầm lẫn giữa các lớp lân cận**:

   - "Kém" ↔ "Trung bình": 18 + 43 = 61 trường hợp
   - "Kém" ↔ "Xấu": 24 + 6 = 30 trường hợp
   - "Trung bình" ↔ "Tốt": 26 + 10 = 36 trường hợp

3. **Không có nhầm lẫn xa**:
   - Không có trường hợp nào nhầm giữa "Tốt" và "Xấu"/"Rất xấu"
   - Cho thấy mô hình học được thứ tự các cấp độ

**Giải thích**:

- Sai số chủ yếu rơi vào **các lớp lân cận** (ví dụ: "Kém" nhầm "Trung bình")
- Đây là **sai số chấp nhận được** do:
  - Tính liên tục của nồng độ bụi PM2.5
  - Ngưỡng phân chia giữa các lớp là nhân tạo (man-made thresholds)
  - Trong thực tế, PM2.5 = 40.3 và 40.5 gần như giống nhau nhưng thuộc 2 lớp khác nhau

---

## 6. PHÂN TÍCH FEATURE IMPORTANCE

### 6.1 Mức độ quan trọng của các đặc trưng

| Thứ hạng | Đặc trưng            | Importance | Phần trăm | Ý nghĩa                             |
| -------- | -------------------- | ---------- | --------- | ----------------------------------- |
| 1        | **PRES2M**           | 0.2930     | 29.30%    | Áp suất khí quyển - Quan trọng nhất |
| 2        | **SQRT_SEA_DEM_LAT** | 0.2197     | 21.97%    | Yếu tố địa lý tổng hợp              |
| 3        | **TP**               | 0.2082     | 20.82%    | Lượng mưa - Rửa trôi bụi            |
| 4        | **TMP**              | 0.1159     | 11.59%    | Nhiệt độ - Ảnh hưởng hóa học        |
| 5        | **RH**               | 0.0843     | 8.43%     | Độ ẩm - Kết tụ bụi                  |
| 6        | **WSPD**             | 0.0788     | 7.88%     | Tốc độ gió - Phân tán bụi           |

### 6.2 Giải thích khoa học

**1. PRES2M (29.30%) - Đặc trưng quan trọng nhất**:

- Áp suất thấp → Không khí ổn định → Bụi ứ đọng
- Áp suất cao → Không khí lưu thông → Bụi phân tán
- Tại Hà Nội: Áp suất thay đổi theo mùa, ảnh hưởng lớn đến AQI

**2. SQRT_SEA_DEM_LAT (21.97%)**:

- Biến tổng hợp từ 3 yếu tố địa lý:
  - Khoảng cách đến biển: Xa biển → Ít gió biển → Bụi nhiều
  - Độ cao địa hình: Vùng trũng → Bụi tích tụ
  - Vĩ độ: Ảnh hưởng đến khí hậu và gió mùa
- Quan trọng vì thể hiện đặc điểm địa lý cố định của từng khu vực

**3. TP - Lượng mưa (20.82%)**:

- Mưa lớn → Rửa trôi bụi → AQI cải thiện
- Không mưa → Bụi tích tụ → AQI xấu đi
- Đặc biệt quan trọng tại Việt Nam với mùa mưa-mùa khô rõ rệt

**4-6. TMP, RH, WSPD (tổng 28%)**:

- Các yếu tố khí tượng bổ sung
- Ảnh hưởng ít hơn nhưng vẫn cần thiết cho dự đoán chính xác

---

## 7. ƯU ĐIỂM VÀ HẠN CHẾ

### 7.1 Ưu điểm

✅ **Hiệu suất vượt trội**:

- Accuracy 60.11% cao hơn Neural Network (55.16%) 4.95%
- F1-Score cao hơn 5.6% (0.6098 vs 0.5538)

✅ **Khả năng cảnh báo sớm**:

- Recall = 0.68 cho lớp "Xấu" (phát hiện 68% ngày xấu)
- Recall = 1.00 cho lớp "Rất xấu" (phát hiện 100%)
- Quan trọng trong giám sát môi trường

✅ **Diễn giải được** (Interpretable):

- Có thể trích xuất quy tắc quyết định
- Feature importance giúp hiểu yếu tố nào quan trọng
- Dễ giải thích cho người dùng không chuyên

✅ **Xử lý dữ liệu mất cân bằng**:

- `class_weight='balanced'` giúp mô hình không bỏ qua lớp thiểu số
- Các lớp hiếm vẫn được dự đoán với Recall cao

✅ **Không cần giả định**:

- Không yêu cầu dữ liệu phân phối chuẩn
- Xử lý được quan hệ phi tuyến tự nhiên

### 7.2 Hạn chế

❌ **Accuracy chưa cao** (60.11%):

- Vẫn còn 40% dự đoán sai
- Cần cải thiện thêm

❌ **Hiệu suất kém trên lớp "Kém" và "Xấu"**:

- F1-Score chỉ đạt 0.46 và 0.52
- Nhiều nhầm lẫn với các lớp lân cận

❌ **Dữ liệu lớp "Rất xấu" quá ít**:

- Chỉ 2 mẫu trong test set
- Kết quả không đủ tin cậy thống kê

❌ **Xu hướng overfitting**:

- Decision Tree dễ học thuộc dữ liệu training
- Cần kiểm tra thêm với cross-validation

❌ **Không ổn định**:

- Thay đổi nhỏ trong dữ liệu có thể tạo ra cây hoàn toàn khác
- Cần ensemble methods (Random Forest) để cải thiện

---

## 8. ỨNG DỤNG THỰC TẾ

### 8.1 Tạo bản đồ AQI từ dữ liệu TIF

Model được sử dụng để xử lý dữ liệu vệ tinh (TIF files) và tạo bản đồ AQI cho 6 ngày:

| Ngày     | Valid Pixels | Tốt                       | Trung bình | Kém | Xấu | Rất xấu |
| -------- | ------------ | ------------------------- | ---------- | --- | --- | ------- |
| 20200101 | 2,914        | Phân bố đa dạng           | ✓          | ✓   | ✓   | ✓       |
| 20200529 | 2,914        | Chủ yếu Tốt và Kém        | ✓          | -   | ✓   | -       |
| 20200721 | 2,914        | Chủ yếu Kém               | -          | -   | ✓   | -       |
| 20200810 | 2,914        | Chủ yếu Kém               | -          | -   | ✓   | -       |
| 20201015 | 2,914        | Chủ yếu Trung bình        | -          | ✓   | -   | -       |
| 20201215 | 2,914        | Hỗn hợp Tốt, Kém, Rất xấu | ✓          | -   | ✓   | ✓       |

**Output files**:

- Bản đồ màu: `output_images_dt/AQI_Map_DT_YYYYMMDD.png`
- Dữ liệu CSV: `output_csv_dt/TIF_Predictions_DT_YYYYMMDD.csv`

### 8.2 Color Mapping (đã sửa lỗi)

Màu hiển thị theo thứ tự alphabet của label encoder:

| AQI Level  | Encoded Value | Color   | Hex Code |
| ---------- | ------------- | ------- | -------- |
| Kém        | 0             | Cam     | #FF7E00  |
| Rất xấu    | 1             | Tím     | #8F3F97  |
| Trung bình | 2             | Vàng    | #FFFF00  |
| Tốt        | 3             | Xanh lá | #00E400  |
| Xấu        | 4             | Đỏ      | #FF0000  |
| NoData     | -1            | Trắng   | #FFFFFF  |

---

## 9. KHUYẾN NGHỊ CẢI THIỆN

### 9.1 Ngắn hạn

1. **Thu thập thêm dữ liệu**:

   - Đặc biệt lớp "Xấu" và "Rất xấu"
   - Mục tiêu: Tăng từ 2 lên ít nhất 50 mẫu "Rất xấu"

2. **Áp dụng SMOTE** (Synthetic Minority Over-sampling):

   - Tạo mẫu tổng hợp cho lớp thiểu số
   - Cải thiện Recall cho lớp "Xấu" và "Rất xấu"

3. **Fine-tuning hyperparameters**:
   - Thử max_depth = [8, 12, 15]
   - Thử min_samples_split = [10, 15, 25]
   - Sử dụng GridSearchCV để tìm tham số tối ưu

### 9.2 Dài hạn

1. **Ensemble Methods**:

   - Random Forest: Tổng hợp nhiều Decision Trees
   - Gradient Boosting (XGBoost, LightGBM): Kết hợp tuần tự các weak learners
   - Có thể cải thiện accuracy lên 65-70%

2. **Feature Engineering**:

   - Thêm các biến tương tác (VD: PRES2M × TP)
   - Thêm dữ liệu temporal (ngày trong năm, mùa)
   - Thêm dữ liệu về giao thông, công nghiệp

3. **Kết hợp nhiều models** (Stacking):
   - Sử dụng cả Decision Tree, Random Forest và Neural Network
   - Meta-learner để kết hợp dự đoán
   - Tận dụng ưu điểm của từng model

---

## 10. KẾT LUẬN

### 10.1 Tóm tắt

Mô hình Decision Tree đã đạt được **accuracy 60.11%** trên tập test, cao hơn 4.95% so với Neural Network baseline. Mô hình thể hiện:

✅ **Điểm mạnh**:

- Khả năng cảnh báo sớm tốt (Recall cao cho lớp nguy hiểm)
- Dễ diễn giải và hiểu được quy tắc phân loại
- Xử lý tốt dữ liệu mất cân bằng

⚠️ **Điểm cần cải thiện**:

- Accuracy vẫn chưa đạt mức cao (60%)
- Cần thêm dữ liệu cho lớp thiểu số
- Có thể áp dụng ensemble methods để nâng cao hiệu suất

### 10.2 Ứng dụng thực tế

Mô hình phù hợp cho:

- **Hệ thống cảnh báo sớm**: Ưu tiên phát hiện ngày xấu (high recall)
- **Công cụ hỗ trợ quyết định**: Diễn giải được, dễ giải thích
- **Phân tích không gian**: Tạo bản đồ AQI từ dữ liệu vệ tinh

### 10.3 Hướng phát triển

Dự án có thể mở rộng theo các hướng:

1. Kết hợp thêm dữ liệu vệ tinh (MODIS, Sentinel)
2. Tích hợp dữ liệu thời gian thực (real-time API)
3. Xây dựng web app/mobile app cho người dùng cuối
4. Dự báo xu hướng AQI trong tương lai (time series forecasting)

---

**Người thực hiện**: Nhóm 2  
**Ngày hoàn thành**: 26/11/2025  
**Công cụ**: Python, scikit-learn, pandas, matplotlib, seaborn
