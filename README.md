# Dự Báo Chất Lượng Không Khí - Decision Tree Model

## Mô tả

Dự án sử dụng **Decision Tree** để dự báo chỉ số chất lượng không khí (AQI) tại Hà Nội dựa trên dữ liệu khí tượng.

**Đầu vào**:

- Dữ liệu khí tượng: PRES2M, RH, WSPD, TMP, TP (5 features)
- Dữ liệu địa hình: SQRT_SEA_DEM_LAT (1 feature)
- **Đặc trưng thời gian** (Feature Engineering): month, month_sin, month_cos, day_of_year (4 features)

**Đầu ra**: Dự báo AQI (5 mức: Tốt, Trung bình, Kém, Xấu, Rất xấu)

**Khu vực**: Hà Nội (miền Bắc Việt Nam)

**Kết quả**: Accuracy **41.42%**, F1-Score **0.43** (Macro F1: **0.32**)

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- pip (Python package manager)

### Cài đặt Dependencies

**Cách 1: Cài đặt thủ công**

```bash
# Core libraries
pip install pandas numpy scikit-learn imbalanced-learn joblib

# Visualization
pip install matplotlib seaborn

# Geospatial (tùy chọn, cho xử lý file TIF)
pip install rasterio

# Jupyter notebook (nếu muốn chạy .ipynb)
pip install jupyter ipykernel
```

### Kiểm tra cài đặt

```bash
python -c "import pandas, numpy, sklearn, joblib; print('Cài đặt thành công!')"
```

---

## Cách chạy Decision Tree

**Option A: Chạy Complete Pipeline (Tất cả trong một)**

```bash
python decision_tree_complete.py
```

Script này sẽ:

- ✅ Tự động load và clean dữ liệu từ `data_onkk_clean.csv`
- ✅ Training model với GridSearchCV + SMOTE
- ✅ Generate tất cả báo cáo (text + markdown)
- ✅ Tạo biểu đồ (confusion matrix, feature importance)
- ✅ Xử lý TIF files và tạo bản đồ AQI

**Thời gian chạy**: ~2-5 phút (tùy cấu hình máy)

**Option B: Chạy Jupyter Notebook**

```bash
# Khởi động Jupyter
jupyter notebook decision_tree_complete.ipynb

# Hoặc sử dụng VS Code (mở file .ipynb và click Run All)
```

**Option C: Chạy từng bước riêng**

```bash
# Bước 1: Clean data (nếu chưa có data_onkk_clean.csv)
python clean_data.py

# Bước 2: Training và evaluation
python decision_tree_complete.py

# Bước 3: Kiểm tra feature importance
python check_feature_importance.py

# Bước 4: Tạo bản đồ AQI (tùy chọn)
python create_aqi_maps.py
```

---

## Feature Engineering

Dữ liệu khí tượng và ô nhiễm mang tính chất **mùa vụ và chu kỳ rất mạnh**. Do đó, nhóm đã trích xuất thêm **4 cột dữ liệu từ cột 'time'** để cải thiện khả năng dự báo:

### Đặc trưng thời gian được trích xuất:

1. **`month`** (Tháng): Quan trọng nhất để nắm bắt được mùa

   - Giá trị: 1-12
   - Giúp model học được pattern theo mùa (Đông/Xuân ô nhiễm cao hơn Hè/Thu)

2. **`month_sin`** và **`month_cos`**: Cyclical encoding của tháng

   - Công thức:
     - `month_sin = sin(2π × month / 12)`
     - `month_cos = cos(2π × month / 12)`
   - **Lợi ích**: Giúp model hiểu rằng **tháng 12 rất gần với tháng 1**
     (điều mà biến 'month' thông thường không thể hiện được)
   - Tháng 1 và tháng 12 có khoảng cách nhỏ trong không gian sin-cos

3. **`day_of_year`**: Ngày trong năm (1-366)
   - Để bắt **trend mịn hơn** trong năm
   - Giúp model học được sự thay đổi liên tục của chất lượng không khí

### Triển khai:

```python
def add_temporal_features(df):
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Extract month
    df['month'] = df['time'].dt.month

    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Day of year
    df['day_of_year'] = df['time'].dt.dayofyear

    # Remove original time column
    df = df.drop('time', axis=1)
    return df
```

**Kết quả**: Tổng số features tăng từ **6 → 10 features**, giúp model học tốt hơn các pattern theo thời gian.

---
