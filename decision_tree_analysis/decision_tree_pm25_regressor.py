# -*- coding: utf-8 -*-
import sys
import io

# Thiết lập encoding UTF-8 cho output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime
import warnings
import os
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

warnings.filterwarnings('ignore')

# Set Vietnamese font
plt.rcParams['font.family'] = 'DejaVu Sans'

print("="*80)
print("DECISION TREE REGRESSOR - DỰ ĐOÁN NỒNG ĐỘ PM2.5")
print("="*80)


# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data(filepath='../data_onkk (2).csv'):
    """Load và preprocess dữ liệu"""
    
    print(f"\n1. Loading data...")
    df = pd.read_csv(filepath)
    
    # Remove missing values
    df_clean = df.dropna()
    
    print(f"   ✓ Loaded {len(df_clean)} samples")
    
    return df_clean


# ============================================================================
# 2. PREPARE FEATURES & TARGET
# ============================================================================

def prepare_data(df):
    """Chuẩn bị features và target"""
    
    print(f"\n2. Preparing data...")
    
    # Features
    feature_columns = ['PRES2M', 'RH', 'TMP', 'TP', 'WSPD', 'SQRT_SEA_DEM_LAT']
    
    X = df[feature_columns]
    y = df['pm25']  # Target là giá trị PM2.5 thực tế (column name lowercase)
    
    print(f"   Features: {feature_columns}")
    print(f"   Target: PM2.5 (continuous values)")
    print(f"   PM2.5 range: {y.min():.2f} - {y.max():.2f} μg/m³")
    print(f"   PM2.5 mean: {y.mean():.2f} ± {y.std():.2f} μg/m³")
    
    return X, y, feature_columns


# ============================================================================
# 3. SPLIT & SCALE DATA
# ============================================================================

def split_and_scale_data(X, y):
    """Chia dữ liệu và chuẩn hóa"""
    
    print(f"\n3. Splitting data (test_size=0.2)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    print(f"\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   ✓ Features scaled")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================================
# 4. TRAIN MODEL
# ============================================================================

def train_model(X_train, y_train):
    """Huấn luyện Decision Tree Regressor"""
    
    print(f"\n5. Training Decision Tree Regressor...")
    
    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    print(f"   Hyperparameters:")
    print(f"     max_depth: 10")
    print(f"     min_samples_split: 20")
    print(f"     min_samples_leaf: 10")
    
    model.fit(X_train, y_train)
    
    print(f"   ✓ Model trained successfully")
    print(f"   Tree depth: {model.get_depth()}")
    print(f"   Number of leaves: {model.get_n_leaves()}")
    
    return model


# ============================================================================
# 5. EVALUATE MODEL
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Đánh giá model"""
    
    print("\n6. Evaluating model...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Train metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n   Train Metrics:")
    print(f"     MSE:  {train_mse:.4f}")
    print(f"     RMSE: {train_rmse:.4f} μg/m³")
    print(f"     MAE:  {train_mae:.4f} μg/m³")
    print(f"     R²:   {train_r2:.4f}")
    
    print(f"\n   Test Metrics:")
    print(f"     MSE:  {test_mse:.4f}")
    print(f"     RMSE: {test_rmse:.4f} μg/m³")
    print(f"     MAE:  {test_mae:.4f} μg/m³")
    print(f"     R²:   {test_r2:.4f}")
    
    # Cross-validation
    print(f"\n   Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                  scoring='r2', n_jobs=-1)
    print(f"     CV R² scores: {cv_scores}")
    print(f"     CV R² mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'cv_scores': cv_scores,
        'y_test_pred': y_test_pred
    }


# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

def plot_feature_importance(model, feature_names, save_path='../output_reports/dt_pm25_feature_importance.png'):
    """Vẽ feature importance"""
    
    print(f"\n7. Plotting feature importance...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importances)))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(importances)), importances[indices], color=colors)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('Decision Tree Regressor - Feature Importance (PM2.5)', fontsize=16, fontweight='bold')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{importances[indices[i]]:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {save_path}")
    plt.close()
    
    print(f"\n   Feature Importance Ranking:")
    for i, idx in enumerate(indices, 1):
        print(f"     {i}. {feature_names[idx]}: {importances[idx]:.4f}")


def plot_predictions(y_test, y_pred, save_path='../output_reports/dt_pm25_predictions.png'):
    """Vẽ scatter plot actual vs predicted"""
    
    print(f"\n8. Plotting predictions...")
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual PM2.5 (μg/m³)', fontsize=12)
    plt.ylabel('Predicted PM2.5 (μg/m³)', fontsize=12)
    plt.title('Decision Tree Regressor - Actual vs Predicted PM2.5', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {save_path}")
    plt.close()


def plot_residuals(y_test, y_pred, save_path='../output_reports/dt_pm25_residuals.png'):
    """Vẽ residual plot"""
    
    print(f"\n9. Plotting residuals...")
    
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted PM2.5 (μg/m³)', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Residuals vs Predicted Values', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residuals Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {save_path}")
    plt.close()


# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

def save_results(results, model, scaler, feature_names):
    """Lưu kết quả và model"""
    
    print(f"\n10. Saving results...")
    
    # Save report
    report_path = '../output_reports/decision_tree_pm25_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DECISION TREE REGRESSOR - DỰ ĐOÁN NỒNG ĐỘ PM2.5\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Ngày đánh giá: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TRAIN METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"MSE:  {results['train_mse']:.4f}\n")
        f.write(f"RMSE: {results['train_rmse']:.4f} μg/m³\n")
        f.write(f"MAE:  {results['train_mae']:.4f} μg/m³\n")
        f.write(f"R²:   {results['train_r2']:.4f}\n\n")
        
        f.write("TEST METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"MSE:  {results['test_mse']:.4f}\n")
        f.write(f"RMSE: {results['test_rmse']:.4f} μg/m³\n")
        f.write(f"MAE:  {results['test_mae']:.4f} μg/m³\n")
        f.write(f"R²:   {results['test_r2']:.4f}\n\n")
        
        f.write("CROSS-VALIDATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"5-fold CV R² mean: {results['cv_scores'].mean():.4f} ± {results['cv_scores'].std():.4f}\n\n")
        
        f.write("FEATURE IMPORTANCE:\n")
        f.write("-" * 80 + "\n")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i, idx in enumerate(indices, 1):
            f.write(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}\n")
    
    print(f"   ✓ Saved report to {report_path}")
    
    # Save model
    model_path = 'decision_tree_pm25_regressor.pkl'
    joblib.dump(model, model_path)
    print(f"   ✓ Saved model to {model_path}")
    
    # Save scaler
    scaler_path = 'decision_tree_pm25_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Saved scaler to {scaler_path}")


# ============================================================================
# 8. GENERATE PM2.5 MAPS
# ============================================================================

def read_tif_file(filepath):
    """Đọc file TIF"""
    try:
        img = Image.open(filepath)
        data = np.array(img, dtype=np.float32)
        return data
    except Exception as e:
        print(f"   ✗ Error reading {filepath}: {e}")
        return None


def scan_available_dates(base_dir='../Feature_Maps-20251116T094941Z-1-001/Feature_Maps'):
    """Quét các ngày có đủ dữ liệu"""
    
    feature_folders = ['PRES2M', 'RH', 'TMP', 'TP', 'WSPD']
    
    all_dates = set()
    for folder in feature_folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
            dates = [f.split('_')[-1].replace('.tif', '') for f in files]
            if not all_dates:
                all_dates = set(dates)
            else:
                all_dates &= set(dates)
    
    return sorted(list(all_dates))


def process_date(date, model, scaler, feature_names, 
                 base_dir='../Feature_Maps-20251116T094941Z-1-001/Feature_Maps',
                 output_dir='../output_images_dt_pm25',
                 csv_dir='../output_csv_dt_pm25'):
    """Xử lý một ngày"""
    
    print(f"\n   Processing date: {date}")
    
    # Load TIF files
    features = {}
    for name in ['PRES2M', 'RH', 'TMP', 'TP', 'WSPD']:
        filepath = os.path.join(base_dir, name, f'{name}_{date}.tif')
        data = read_tif_file(filepath)
        if data is None:
            return False
        features[name] = data
    
    # Load SQRT_SEA_DEM_LAT
    sqrt_path = os.path.join(base_dir, 'SQRT_SEA_DEM_LAT.tif')
    sqrt_data = read_tif_file(sqrt_path)
    
    # Get shape
    shape = features['PRES2M'].shape
    total_pixels = shape[0] * shape[1]
    
    # Flatten
    flattened = {}
    for name, data in features.items():
        flattened[name] = data.flatten()
    
    if sqrt_data is not None:
        flattened['SQRT_SEA_DEM_LAT'] = sqrt_data.flatten()
    else:
        flattened['SQRT_SEA_DEM_LAT'] = np.full(total_pixels, 3.5)
    
    # Create DataFrame (only use 6 features, no SQRT_LAT_LON)
    X_df = pd.DataFrame(flattened)
    
    # Valid mask
    valid_mask = np.ones(total_pixels, dtype=bool)
    for name in ['PRES2M', 'RH', 'TMP', 'TP', 'WSPD']:
        valid_mask &= (flattened[name] != -9999)
    
    # Predict
    predictions = np.full(total_pixels, -9999.0, dtype=np.float32)
    
    X_valid = X_df[valid_mask].values
    if len(X_valid) > 0:
        X_scaled = scaler.transform(X_valid)
        y_pred = model.predict(X_scaled)
        predictions[valid_mask] = y_pred
    
    # Create map
    predictions_2d = predictions.reshape(shape)
    
    # Custom colormap for PM2.5
    colors_list = [
        '#00E400',  # Tốt (0-15.4)
        '#FFFF00',  # Trung bình (15.4-40.4)
        '#FF7E00',  # Kém (40.4-65.4)
        '#FF0000',  # Xấu (65.4-150.4)
        '#8F3F97',  # Rất xấu (>150.4)
    ]
    cmap = LinearSegmentedColormap.from_list('pm25', colors_list, N=256)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Mask invalid data
    masked_data = np.ma.masked_where(predictions_2d == -9999, predictions_2d)
    
    im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=150, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PM2.5 (μg/m³)', fontsize=12)
    
    # Add AQI level lines
    cbar.ax.axhline(y=15.4, color='black', linewidth=1, linestyle='--', alpha=0.5)
    cbar.ax.axhline(y=40.4, color='black', linewidth=1, linestyle='--', alpha=0.5)
    cbar.ax.axhline(y=65.4, color='black', linewidth=1, linestyle='--', alpha=0.5)
    cbar.ax.axhline(y=150.4, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    ax.set_title(f'Decision Tree - PM2.5 Concentration Map {date}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save map
    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, f'PM25_Map_DT_{date}.png')
    plt.savefig(map_path, dpi=150, bbox_inches='tight')
    print(f"     ✓ Map saved to {map_path}")
    plt.close()
    
    # Save CSV
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'PM25_Predictions_DT_{date}.csv')
    
    predictions_1d = predictions_2d.flatten()
    valid_count = np.sum(predictions_1d != -9999)
    
    df_output = pd.DataFrame({
        'Pixel_Index': range(len(predictions_1d)),
        'PM2.5': predictions_1d
    })
    
    df_output.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"     ✓ CSV saved to {csv_path}")
    print(f"     Valid pixels: {valid_count:,}/{total_pixels:,}")
    
    return True


def generate_pm25_maps(model, scaler, feature_names):
    """Tạo bản đồ PM2.5 cho tất cả các ngày"""
    
    print("\n" + "="*80)
    print("TẠO BẢN ĐỒ NỒNG ĐỘ PM2.5")
    print("="*80)
    
    print("\n11. Scanning for available dates...")
    dates = scan_available_dates()
    print(f"   ✓ Found {len(dates)} dates with complete data")
    
    print(f"\n   Processing {len(dates)} dates...")
    
    success_count = 0
    for date in dates:
        if process_date(date, model, scaler, feature_names):
            success_count += 1
    
    print(f"\n   ✓ Successfully processed {success_count}/{len(dates)} dates")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load data
    df = load_data()
    
    # Prepare data
    X, y, feature_names = prepare_data(df)
    
    # Split & scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Visualizations
    plot_feature_importance(model, feature_names)
    plot_predictions(y_test, results['y_test_pred'])
    plot_residuals(y_test, results['y_test_pred'])
    
    # Save
    save_results(results, model, scaler, feature_names)
    
    # Generate PM2.5 maps
    generate_pm25_maps(model, scaler, feature_names)
    
    print("\n" + "="*80)
    print("HOÀN TẤT ✓")
    print("="*80)


if __name__ == '__main__':
    main()
