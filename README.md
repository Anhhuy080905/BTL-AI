# Dự Báo Chất Lượng Không Khí Hà Nội - AQI Classification

## Cài đặt

```bash
pip install torch numpy pandas scikit-learn pillow matplotlib joblib
```

## Chạy Model

1. Mở notebook:

```bash
jupyter notebook onkk-model-test.ipynb
```

2. Chạy tất cả cells (Run All)

## Output

- **Báo cáo đánh giá**: `output_reports/classification_report_notebook.txt`
- **Bản đồ AQI**: `output_images/AQI_Map_*.png`
- **CSV predictions**: `output_csv/TIF_Predictions_*.csv`

## Kết quả

- Accuracy: **55.16%**
- Precision: **0.6479**
- Recall: **0.5516**
- F1-Score: **0.5538**
