# Dự Báo Chất Lượng Không Khí Hà Nội - AQI Classification

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
