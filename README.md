# Dự Báo Chất Lượng Không Khí - Decision Tree Model

## Mô tả

Dự án sử dụng **Decision Tree** để dự báo chỉ số chất lượng không khí (AQI) tại Hà Nội dựa trên dữ liệu khí tượng.

**Đầu vào**: Dữ liệu khí tượng (PRES2M, RH, WSPD, TMP, TP) và địa hình (SQRT_SEA_DEM_LAT)

**Đầu ra**: Dự báo AQI (5 mức: Tốt, Trung bình, Kém, Xấu, Rất xấu)

**Khu vực**: Hà Nội (miền Bắc Việt Nam)

**Kết quả**: Accuracy **33.87%**, F1-Score **0.37** (Macro F1: **0.24**)

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- pip (Python package manager)

### Cài đặt Dependencies

**Cách 1: Cài đặt từ requirements.txt (Khuyến nghị)**

```bash
pip install -r requirements.txt
```

**Cách 2: Cài đặt thủ công**

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

**Kết quả:**

- Accuracy: **33.87%**
- Weighted Precision: **0.44**
- Weighted Recall: **0.34**
- Weighted F1-Score: **0.37**
- Macro F1-Score: **0.24**

**Output Files:**

```
output_reports/
├── decision_tree_report.txt              # Báo cáo text chi tiết
├── decision_tree_summary.md              # Báo cáo markdown tổng hợp
├── decision_tree_confusion_matrix.png    # Ma trận nhầm lẫn
└── decision_tree_feature_importance.png  # Biểu đồ tầm quan trọng features

output_images_dt/
└── AQI_Map_DT_*.png                      # Bản đồ dự báo AQI theo ngày

output_csv_dt/
└── TIF_Predictions_DT_*.csv              # Dữ liệu dự báo dạng CSV

model/ (hoặc thư mục gốc)
├── decision_tree_classifier.pkl           # Model đã training
├── decision_tree_scaler.pkl              # StandardScaler
└── decision_tree_label_encoder.pkl       # LabelEncoder
```

---

## Troubleshooting

### Lỗi thường gặp

**1. ModuleNotFoundError: No module named 'xxx'**

```bash
# Cài đặt lại dependencies
pip install -r requirements.txt
```

**2. FileNotFoundError: data_onkk_clean.csv**

```bash
# Chạy script clean data trước
python clean_data.py
```

**3. Lỗi encoding khi chạy trên Windows**

Script đã được cấu hình UTF-8 tự động. Nếu vẫn gặp lỗi:

```bash
# Chạy với encoding UTF-8
chcp 65001
python decision_tree_complete.py
```

**4. Rasterio không cài đặt được (Windows)**

Rasterio là tùy chọn (optional) cho xử lý TIF files. Nếu gặp lỗi:

```bash
# Bỏ qua rasterio, vẫn chạy được model
pip install --no-deps rasterio

# Hoặc download wheel từ: https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio
pip install rasterio‑xxx.whl
```

**5. Out of Memory khi training**

```python
# Giảm kích thước GridSearch trong code
param_grid = {
    'max_depth': [10, 15],  # Giảm từ [5, 10, 15, 20]
    'min_samples_split': [5, 10],  # Giảm từ [2, 5, 10]
}
```

### Kiểm tra kết quả nhanh

```bash
# Xem accuracy và metrics
cat output_reports/decision_tree_report.txt

# Xem confusion matrix
start output_reports/decision_tree_confusion_matrix.png  # Windows
# open output_reports/decision_tree_confusion_matrix.png  # macOS
# xdg-open output_reports/decision_tree_confusion_matrix.png  # Linux
```

---

---

## Tài liệu tham khảo

- [DECISION_TREE_COMPLETE_GUIDE.md](DECISION_TREE_COMPLETE_GUIDE.md) - Hướng dẫn chi tiết
- [TIME_SERIES_SPLIT_RESULTS.md](TIME_SERIES_SPLIT_RESULTS.md) - Kết quả Time Series validation
- [decision_tree_summary.md](output_reports/decision_tree_summary.md) - Báo cáo đầy đủ
