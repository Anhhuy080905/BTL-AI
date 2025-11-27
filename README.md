# Dự Báo Chất Lượng Không Khí - Miền Bắc Việt Nam (Hà Nội)

## Bài toán

**Đầu vào**: Dữ liệu khí tượng (PRES2M, RH, WSPD, TMP, TP) và địa hình (SQRT_SEA_DEM_LAT)

**Đầu ra**: Dự báo AQI và nồng độ PM2.5

**Khu vực**: Hà Nội (đại diện cho miền Bắc Việt Nam)

**Phương pháp**:
1. Tính chỉ số AQI dựa trên nồng độ PM2.5 từ trạm quan trắc
2. Thử nghiệm 4 mô hình học máy (Neural Network & Decision Tree cho Classification & Regression)
3. Đánh giá kết quả trên các chỉ số: Accuracy, Precision, Recall, F1, RMSE, MAE, R²
4. Đề xuất mô hình tốt nhất cho từng task
5. Ứng dụng tạo bản đồ dự báo PM2.5 và AQI
6. Hiển thị bản đồ với color mapping theo chuẩn AQI

**Lưu ý**: Mô hình hiện tại dự đoán AQI/PM2.5 cho **cùng ngày** dựa trên dữ liệu khí tượng. Để dự báo nhiều ngày tiếp theo, cần bổ sung mô hình time series (LSTM/GRU).

---

## Cài đặt

```bash
pip install torch numpy pandas scikit-learn pillow matplotlib joblib seaborn
```

## Chạy Models

### 1. Neural Network Model (Baseline)

**Option A: Chạy Notebook**

```bash
jupyter notebook model/onkk-model-test.ipynb
```

**Kết quả:**

- Accuracy: **55.16%**
- Precision: **0.6479**
- Recall: **0.5516**
- F1-Score: **0.5538**

**Output:**

- Báo cáo: `output_reports/classification_report_notebook.txt`
- Bản đồ AQI: `output_images/AQI_Map_*.png`
- CSV predictions: `output_csv/TIF_Predictions_*.csv`

---

### 2. Decision Tree Model (Better Performance)

**Option A: Chạy Python Script**

```bash
cd decision_tree_analysis
python decision_tree_model.py
```

**Option B: Chạy Notebook trong VS Code**

Mở file `decision_tree_analysis/decision-tree-model.ipynb` trong VS Code và click **Run All** hoặc chạy từng cell bằng **Shift+Enter**.

_Lưu ý: VS Code hỗ trợ chạy Jupyter notebook trực tiếp, không cần cài đặt Jupyter server riêng._

**Kết quả:**

- Accuracy: **60.11%** (tốt hơn Neural Network 4.95%)
- Precision: **0.6449**
- Recall: **0.6011**
- F1-Score: **0.6098**

**Feature Importance:**

1. PRES2M (Áp suất): 29.30%
2. SQRT_SEA_DEM_LAT: 21.97%
3. TP (Lượng mưa): 20.82%

**Output:**

- Model files: `decision_tree_analysis/decision_tree_*.pkl`
- Báo cáo: `output_reports/decision_tree_report.txt`
- Confusion Matrix: `output_reports/decision_tree_confusion_matrix.png`
- Feature Importance: `output_reports/decision_tree_feature_importance.png`
- Bản đồ AQI: `output_images_dt/AQI_Map_DT_*.png`
- CSV predictions: `output_csv_dt/TIF_Predictions_DT_*.csv`

---

### 3. PM2.5 Regression Model (Neural Network)

**Chạy Notebook:**

```bash
jupyter notebook pm25_analysis/pm25-regression-analysis.ipynb
```

**Kết quả:**

- Test RMSE: **13.28 μg/m³**
- Test MAE: **8.67 μg/m³**
- Test R²: **0.7234**

**Output:**

- Model: `pm25_analysis/pm25_regressor.pth`
- Báo cáo: `output_reports/pm25_regression_report.txt`
- Bản đồ PM2.5: `output_images_pm25/PM25_Map_*.png`
- CSV predictions: `output_csv_pm25/TIF_Predictions_PM25_*.csv`

---

### 4. Decision Tree PM2.5 Regressor (Interpretable)

**Chạy Python Script:**

```bash
cd decision_tree_analysis
python decision_tree_pm25_regressor.py
```

**Kết quả:**

- Test RMSE: **18.76 μg/m³**
- Test MAE: **12.05 μg/m³**
- Test R²: **0.5143**
- 5-fold CV R²: **0.4732 ± 0.0974**

**Feature Importance:**

1. PRES2M (Áp suất): 33.40%
2. WSPD (Tốc độ gió): 19.84%
3. TP (Lượng mưa): 15.21%

**Output:**

- Model files: `decision_tree_analysis/decision_tree_pm25_*.pkl`
- Báo cáo chi tiết: `output_reports/decision_tree_pm25_summary.md`
- Báo cáo text: `output_reports/decision_tree_pm25_report.txt`
- Visualizations:
  - Feature Importance: `output_reports/dt_pm25_feature_importance.png`
  - Predictions: `output_reports/dt_pm25_predictions.png`
  - Residuals: `output_reports/dt_pm25_residuals.png`
- Bản đồ PM2.5: `output_images_dt_pm25/PM25_Map_DT_*.png`
- CSV predictions: `output_csv_dt_pm25/PM25_Predictions_DT_*.csv`

---

## So sánh Models

### AQI Classification (5 classes)

| Model              | Accuracy   | Precision | Recall | F1-Score | Ưu điểm                                |
| ------------------ | ---------- | --------- | ------ | -------- | -------------------------------------- |
| Neural Network     | 55.16%     | 0.6479    | 0.5516 | 0.5538   | Học được pattern phức tạp              |
| **Decision Tree**  | **60.11%** | **0.6449**| **0.6011** | **0.6098** | **Dễ diễn giải, Recall cao lớp nguy hiểm** |

**Kết luận**: Decision Tree tốt hơn (+4.95% accuracy), đặc biệt phù hợp cho hệ thống cảnh báo sớm.

### PM2.5 Regression (continuous values)

| Model                 | Test RMSE      | Test MAE       | Test R²        | Ưu điểm                           |
| --------------------- | -------------- | -------------- | -------------- | --------------------------------- |
| **Neural Network**    | **13.28**      | **8.67**       | **0.7234**     | **Độ chính xác cao nhất**         |
| Decision Tree         | 18.76          | 12.05          | 0.5143         | Dễ diễn giải, trích xuất quy tắc  |

**Kết luận**: Neural Network vượt trội về độ chính xác (R²=0.72), phù hợp cho dự báo PM2.5 chính xác.

---

## Đề xuất Mô hình

**Cho AQI Classification (5 mức: Tốt, Trung bình, Kém, Xấu, Rất xấu)**:
- ✅ **Decision Tree** (60.11% accuracy)
- Lý do: Recall cao cho lớp nguy hiểm (Xấu: 68%, Rất xấu: 100%), dễ diễn giải quy tắc

**Cho PM2.5 Regression (nồng độ μg/m³)**:
- ✅ **Neural Network** (R²=0.72, RMSE=13.28)
- Lý do: Độ chính xác cao nhất, phù hợp cho dự báo số liệu chính xác

**Cải thiện tương lai**:
- 🔄 Thêm mô hình **time series** (LSTM/GRU) để dự báo 1-7 ngày tiếp theo
- 🔄 Mở rộng khu vực: Hải Phòng, Quảng Ninh, Thái Nguyên
- 🔄 Ensemble methods: Random Forest, XGBoost để tăng accuracy lên 65-70%

---

## Báo cáo Chi tiết

- **Decision Tree AQI**: `output_reports/decision_tree_summary.md`
- **Decision Tree PM2.5**: `output_reports/decision_tree_pm25_summary.md`
- **Neural Network**: `output_reports/classification_report_notebook.txt`
- **PM2.5 Regression**: `output_reports/pm25_regression_report.txt`
