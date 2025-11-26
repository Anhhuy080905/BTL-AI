"""
Decision Tree Model - So sanh hieu nang voi Neural Network
Huan luyen va danh gia Decision Tree cho bai toan phan loai AQI
Author: 23020540 - Nguyen Anh Huy
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================================
# 1. LOAD VÀ CHUẨN BỊ DỮ LIỆU
# ============================================================================

def load_and_prepare_data(data_path='../data_onkk (2).csv'):
    """Load và chuẩn bị dữ liệu"""
    
    print("="*80)
    print("DECISION TREE MODEL - DU BAO AQI")
    print("="*80)
    
    print("\n1. Loading data...")
    df = pd.read_csv(data_path)
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Features và target
    feature_columns = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP', 'SQRT_SEA_DEM_LAT']
    X = df[feature_columns].values
    
    # Chuyển PM2.5 thành AQI categories
    def pm25_to_aqi(pm25):
        if pm25 <= 15.4:
            return 'Tốt'
        elif pm25 <= 40.4:
            return 'Trung bình'
        elif pm25 <= 65.4:
            return 'Kém'
        elif pm25 <= 150.4:
            return 'Xấu'
        else:
            return 'Rất xấu'
    
    y = df['pm25'].apply(pm25_to_aqi).values
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n2. Data info:")
    print(f"   Features: {feature_columns}")
    print(f"   Target classes: {list(label_encoder.classes_)}")
    print(f"   Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"     {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y_encoded, label_encoder, feature_columns


# ============================================================================
# 2. TRAIN/TEST SPLIT
# ============================================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """Chia train/test set"""
    
    print(f"\n3. Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# 3. CHUẨN HÓA DỮ LIỆU
# ============================================================================

def scale_features(X_train, X_test):
    """Chuẩn hóa features"""
    
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   ✓ Features scaled")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# 4. HUẤN LUYỆN DECISION TREE
# ============================================================================

def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """Huấn luyện Decision Tree Classifier"""
    
    print("\n5. Training Decision Tree...")
    print(f"   Hyperparameters:")
    print(f"     max_depth: {max_depth}")
    print(f"     min_samples_split: {min_samples_split}")
    print(f"     min_samples_leaf: {min_samples_leaf}")
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight='balanced'  # Xử lý imbalanced data
    )
    
    model.fit(X_train, y_train)
    
    print(f"   ✓ Model trained successfully")
    print(f"   Tree depth: {model.get_depth()}")
    print(f"   Number of leaves: {model.get_n_leaves()}")
    
    return model


# ============================================================================
# 5. ĐÁNH GIÁ MODEL
# ============================================================================

def evaluate_model(model, X_test, y_test, label_encoder):
    """Đánh giá model trên test set"""
    
    print("\n6. Evaluating model on test set...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n   Overall Metrics:")
    print(f"     Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"     Precision: {precision:.4f}")
    print(f"     Recall:    {recall:.4f}")
    print(f"     F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification Report
    class_names = label_encoder.classes_
    report = classification_report(
        y_test, y_pred, 
        target_names=class_names, 
        zero_division=0
    )
    
    print(f"\n   Classification Report:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'report': report
    }


# ============================================================================
# 6. VISUALIZE RESULTS
# ============================================================================

def plot_confusion_matrix(cm, label_encoder, save_path='../output_reports/decision_tree_confusion_matrix.png'):
    """Vẽ confusion matrix"""
    
    print(f"\n7. Plotting confusion matrix...")
    
    plt.figure(figsize=(10, 8))
    
    # Logical order
    logical_order = ['Tốt', 'Trung bình', 'Kém', 'Xấu', 'Rất xấu']
    
    # Get label indices in logical order
    label_indices = [list(label_encoder.classes_).index(label) for label in logical_order]
    
    # Reorder confusion matrix
    cm_reordered = cm[np.ix_(label_indices, label_indices)]
    
    # Tính phần trăm
    cm_percent = cm_reordered.astype('float') / cm_reordered.sum(axis=1)[:, np.newaxis] * 100
    
    # Annotations
    annot = np.empty_like(cm_reordered).astype(str)
    for i in range(cm_reordered.shape[0]):
        for j in range(cm_reordered.shape[1]):
            annot[i, j] = f'{cm_reordered[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(
        cm_reordered, annot=annot, fmt='', cmap='Blues',
        xticklabels=logical_order,
        yticklabels=logical_order,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Decision Tree - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, save_path='../output_reports/decision_tree_feature_importance.png'):
    """Vẽ feature importance"""
    
    print(f"\n8. Plotting feature importance...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], color='steelblue')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('Decision Tree - Feature Importance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {save_path}")
    plt.close()
    
    print(f"\n   Feature Importance Ranking:")
    for i, idx in enumerate(indices, 1):
        print(f"     {i}. {feature_names[idx]}: {importances[idx]:.4f}")


# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

def save_results(results, model, scaler, label_encoder):
    """Lưu results và model"""
    
    print(f"\n9. Saving results...")
    
    # Save report
    report_path = '../output_reports/decision_tree_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DECISION TREE MODEL - ĐÁNH GIÁ CHI TIẾT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Ngày đánh giá: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f}\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 80 + "\n")
        f.write(results['report'])
        f.write("\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 80 + "\n")
        cm = results['confusion_matrix']
        classes = label_encoder.classes_
        
        # Header
        f.write(f"{'Actual \\ Predicted':<20}")
        for cls in classes:
            f.write(f"{cls:<15}")
        f.write("\n")
        f.write("-" * 80 + "\n")
        
        # Rows
        for i, cls in enumerate(classes):
            f.write(f"{cls:<20}")
            for j in range(len(classes)):
                f.write(f"{cm[i, j]:<15}")
            f.write("\n")
    
    print(f"   ✓ Saved report to {report_path}")
    
    # Save model
    model_path = 'decision_tree_classifier.pkl'
    joblib.dump(model, model_path)
    print(f"   ✓ Saved model to {model_path}")
    
    # Save scaler (nếu cần dùng lại)
    scaler_path = 'decision_tree_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Saved scaler to {scaler_path}")


# ============================================================================
# 8. SO SÁNH VỚI NEURAL NETWORK
# ============================================================================

def compare_with_nn():
    """So sánh với Neural Network"""
    
    print("\n" + "="*80)
    print("SO SÁNH DECISION TREE VS NEURAL NETWORK")
    print("="*80)
    
    # Load NN results từ file
    try:
        with open('../output_reports/classification_report_notebook.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Tìm accuracy của NN
        import re
        nn_accuracy_match = re.search(r'Accuracy:\s+([\d.]+)', content)
        if nn_accuracy_match:
            nn_accuracy = float(nn_accuracy_match.group(1))
            print(f"\nNeural Network Accuracy: {nn_accuracy:.4f} ({nn_accuracy*100:.2f}%)")
        else:
            print("\nKhông tìm thấy accuracy của Neural Network")
            
    except FileNotFoundError:
        print("\nChưa có kết quả Neural Network để so sánh")


# ============================================================================
# 9. XỬ LÝ TIF VÀ TẠO BẢN ĐỒ
# ============================================================================

def read_tif_file(filepath):
    """Đọc file TIF"""
    from PIL import Image
    try:
        img = Image.open(filepath)
        data = np.array(img, dtype=np.float32)
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def find_available_dates(feature_maps_dir='../Feature_Maps-20251116T094941Z-1-001/Feature_Maps'):
    """Tìm các ngày có đầy đủ dữ liệu TIF"""
    
    print("\n10. Scanning for available dates...")
    
    required_features = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP']
    sample_feature_dir = f"{feature_maps_dir}/{required_features[0]}"
    
    import os
    if not os.path.exists(sample_feature_dir):
        print(f"   ❌ Directory not found: {sample_feature_dir}")
        return []
    
    tif_files = [f for f in os.listdir(sample_feature_dir) if f.endswith('.tif')]
    dates = [f.split('_')[1].replace('.tif', '') for f in tif_files]
    
    valid_dates = []
    for date in dates:
        all_exist = True
        for feature in required_features:
            feature_path = f"{feature_maps_dir}/{feature}/{feature}_{date}.tif"
            if not os.path.exists(feature_path):
                all_exist = False
                break
        if all_exist:
            valid_dates.append(date)
    
    valid_dates.sort()
    print(f"   ✓ Found {len(valid_dates)} dates with complete data")
    
    return valid_dates


def process_tif_to_predictions(model, scaler, label_encoder, date, feature_maps_dir='../Feature_Maps-20251116T094941Z-1-001/Feature_Maps'):
    """Xử lý TIF và tạo predictions cho một ngày"""
    
    import os
    from PIL import Image
    from matplotlib.colors import ListedColormap
    
    print(f"\n   Processing date: {date}")
    
    # Load features
    features = {}
    feature_names = ['PRES2M', 'RH', 'WSPD', 'TMP', 'TP']
    
    for feature_name in feature_names:
        filepath = f"{feature_maps_dir}/{feature_name}/{feature_name}_{date}.tif"
        data = read_tif_file(filepath)
        if data is None:
            return None
        features[feature_name] = data
    
    # Load SQRT_SEA_DEM_LAT
    sqrt_path = f"{feature_maps_dir}/SQRT_SEA_DEM_LAT.tif"
    sqrt_data = read_tif_file(sqrt_path)
    
    # Prepare data
    shape = features['PRES2M'].shape
    rows, cols = shape
    total_pixels = rows * cols
    
    # Flatten
    flattened = {}
    for name, data in features.items():
        flattened[name] = data.flatten()
    
    if sqrt_data is not None:
        flattened['SQRT_SEA_DEM_LAT'] = sqrt_data.flatten()
    else:
        flattened['SQRT_SEA_DEM_LAT'] = np.full(total_pixels, 3.5)
    
    # Create DataFrame
    X_df = pd.DataFrame(flattened)
    
    # Valid mask
    valid_mask = np.ones(total_pixels, dtype=bool)
    for name in feature_names:
        valid_mask &= (flattened[name] != -9999)
    
    # Predict
    predictions = np.full(total_pixels, -1, dtype=np.int32)
    predictions_labels = np.full(total_pixels, 'NoData', dtype=object)
    
    X_valid = X_df[valid_mask].values
    if len(X_valid) > 0:
        X_scaled = scaler.transform(X_valid)
        y_pred = model.predict(X_scaled)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        
        predictions[valid_mask] = y_pred
        predictions_labels[valid_mask] = y_pred_labels
    
    # Create map
    predictions_2d = predictions.reshape(shape)
    
    # Label encoder sắp xếp theo alphabet: ['Kém', 'Rất xấu', 'Trung bình', 'Tốt', 'Xấu']
    colors = [
        '#FFFFFF',  # -1: NoData
        '#FF7E00',  # 0: Kém
        '#8F3F97',  # 1: Rất xấu
        '#FFFF00',  # 2: Trung bình
        '#00E400',  # 3: Tốt
        '#FF0000',  # 4: Xấu
    ]
    
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(predictions_2d, cmap=cmap, vmin=-1, vmax=4, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(['NoData', 'Kém', 'Rất xấu', 'Trung bình', 'Tốt', 'Xấu'])
    
    ax.set_title(f'Decision Tree - AQI Map {date}', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('../output_images_dt', exist_ok=True)
    map_path = f'../output_images_dt/AQI_Map_DT_{date}.png'
    plt.savefig(map_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save CSV
    os.makedirs('../output_csv_dt', exist_ok=True)
    csv_path = f'../output_csv_dt/TIF_Predictions_DT_{date}.csv'
    X_df['Predicted_AQI'] = predictions_labels
    X_df['Valid'] = valid_mask
    X_df.to_csv(csv_path, index=False)
    
    print(f"     ✓ Map saved to {map_path}")
    print(f"     ✓ CSV saved to {csv_path}")
    print(f"     Valid pixels: {valid_mask.sum():,}/{total_pixels:,}")
    
    return map_path


# ============================================================================
# 10. MAIN FUNCTION
# ============================================================================

def main():
    """Main function"""
    
    # Load data
    X, y, label_encoder, feature_names = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    model = train_decision_tree(
        X_train_scaled, y_train,
        max_depth=10,  # Giới hạn độ sâu để tránh overfitting
        min_samples_split=20,
        min_samples_leaf=10
    )
    
    # Evaluate
    results = evaluate_model(model, X_test_scaled, y_test, label_encoder)
    
    # Visualize
    plot_confusion_matrix(results['confusion_matrix'], label_encoder)
    plot_feature_importance(model, feature_names)
    
    # Save
    save_results(results, model, scaler, label_encoder)
    
    # Compare
    compare_with_nn()
    
    # Process TIF files
    print("\n" + "="*80)
    print("XỬ LÝ TIF VÀ TẠO BẢN ĐỒ AQI")
    print("="*80)
    
    dates = find_available_dates()
    
    if dates:
        print(f"\n   Processing {len(dates)} dates...")
        for date in dates:
            process_tif_to_predictions(model, scaler, label_encoder, date)
    
    print("\n" + "="*80)
    print("HOÀN TẤT ✓")
    print("="*80)


if __name__ == "__main__":
    main()
