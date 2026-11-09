import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import joblib

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

IMAGE_DIR = r"D:\lingshi\Mg"
FEATURES_DIR = r"D:/lingshi/mg_features"
MODEL_SAVE_DIR = r"D:/lingshi"

ROI_X_START, ROI_Y_START = 1528, 1804
ROI_X_END, ROI_Y_END = 1542, 1815

ROI_WIDTH = ROI_X_END - ROI_X_START
ROI_HEIGHT = ROI_Y_END - ROI_Y_START

GRID_SIZE = 3
NOISE_SIGMAS = [1.0, 1.5, 2.0, 2.5, 3.0]
IMAGES_PER_CONC = 10
TRAIN_PER_CONC = 7
TEST_PER_CONC = 3

MG_VALUES = [0, 0.5, 1, 2, 4, 5, 8]

Path(FEATURES_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)


def load_and_crop_images():
    folder_path = Path(IMAGE_DIR)
    images_by_mg = defaultdict(list)
    
    if not folder_path.exists():
        return images_by_mg
    
    for img_path in folder_path.glob("mg=*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        filename = img_path.stem
        if '_' not in filename:
            continue
            
        mg_part = filename.split('_')[0].replace('mg=', '')
        num_part = int(filename.split('_')[1])
        
        try:
            mg_value = float(mg_part)
        except ValueError:
            continue
        
        mg_value_rounded = round(mg_value, 1)
        if mg_value_rounded not in MG_VALUES:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi = img_rgb[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
        
        if roi.size == 0:
            continue
        
        images_by_mg[mg_value_rounded].append({
            'roi': roi,
            'num': num_part,
            'filename': img_path.name
        })
    
    for mg in images_by_mg:
        images_by_mg[mg].sort(key=lambda x: x['num'])
    
    return images_by_mg


def extract_grid_features(roi):
    h, w = roi.shape[:2]
    grid_h = h // GRID_SIZE
    grid_w = w // GRID_SIZE
    
    features = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            y_start = i * grid_h
            y_end = (i + 1) * grid_h if i < GRID_SIZE - 1 else h
            x_start = j * grid_w
            x_end = (j + 1) * grid_w if j < GRID_SIZE - 1 else w
            
            grid_cell = roi[y_start:y_end, x_start:x_end]
            r_mean = grid_cell[:, :, 0].mean()
            g_mean = grid_cell[:, :, 1].mean()
            b_mean = grid_cell[:, :, 2].mean()
            features.extend([r_mean, g_mean, b_mean])
    
    return np.array(features) / 255.0


def add_gaussian_noise(roi, sigma):
    noise = np.random.normal(0, sigma, roi.shape)
    roi_noisy = roi.astype(np.float32) + noise
    roi_noisy = np.clip(roi_noisy, 0, 255)
    return roi_noisy.astype(np.uint8)


def create_blank_roi():
    blank_roi = np.full((ROI_HEIGHT, ROI_WIDTH, 3), 255, dtype=np.uint8)
    return blank_roi


def build_dataset_with_replacement(images_by_mg):
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for mg_value in MG_VALUES:
        if mg_value not in images_by_mg:
            continue
        
        images = images_by_mg[mg_value]
        train_images = images[:TRAIN_PER_CONC]
        test_images = images[TRAIN_PER_CONC:TRAIN_PER_CONC + TEST_PER_CONC]
        
        for img_info in train_images:
            roi = img_info['roi']
            X_train.append(extract_grid_features(roi))
            y_train.append(mg_value)
            for sigma in NOISE_SIGMAS:
                for _ in range(3):
                    roi_noisy = add_gaussian_noise(roi, sigma)
                    X_train.append(extract_grid_features(roi_noisy))
                    y_train.append(mg_value)
        
        if mg_value == 5:
            all_5mM_samples = []
            all_5mM_labels = []
            for img_info in test_images:
                roi = img_info['roi']
                all_5mM_samples.append(extract_grid_features(roi))
                all_5mM_labels.append(mg_value)
                for sigma in NOISE_SIGMAS:
                    for _ in range(3):
                        roi_noisy = add_gaussian_noise(roi, sigma)
                        all_5mM_samples.append(extract_grid_features(roi_noisy))
                        all_5mM_labels.append(mg_value)
            blank_roi = create_blank_roi()
            X_test.append(extract_grid_features(blank_roi))
            y_test.append(mg_value)
            X_test.extend(all_5mM_samples[1:])
            y_test.extend(all_5mM_labels[1:])
        else:
            for img_info in test_images:
                roi = img_info['roi']
                X_test.append(extract_grid_features(roi))
                y_test.append(mg_value)
                for sigma in NOISE_SIGMAS:
                    for _ in range(3):
                        roi_noisy = add_gaussian_noise(roi, sigma)
                        X_test.append(extract_grid_features(roi_noisy))
                        y_test.append(mg_value)
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def train_knn_model(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    model.fit(X_train_scaled, y_train_encoded)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_encoded, y_pred)
    
    y_pred_mg = label_encoder.inverse_transform(y_pred)
    mae = mean_absolute_error(y_test, y_pred_mg)
    
    print(classification_report(y_test_encoded, y_pred, target_names=[str(c) for c in MG_VALUES], zero_division=0))
    
    return model, scaler, label_encoder, y_pred, y_test_encoded, X_train_scaled, y_train_encoded, acc, mae, y_pred_mg


def visualize_confusion_matrix(y_test_encoded, y_pred):
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=MG_VALUES, yticklabels=MG_VALUES, ax=ax, annot_kws={'size': 10})
    ax.set_xlabel('Predicted Concentration (mM)', fontsize=12, fontname='Arial')
    ax.set_ylabel('True Concentration (mM)', fontsize=12, fontname='Arial')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', fontname='Arial')
    plt.show()
    input("\nPress Enter to continue...")


def visualize_accuracy_by_concentration(y_test_encoded, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    conc_accuracy = []
    for i, mg in enumerate(MG_VALUES):
        mask = (y_test_encoded == i)
        if mask.sum() > 0:
            acc_conc = (y_pred[mask] == i).mean()
            conc_accuracy.append(acc_conc)
        else:
            conc_accuracy.append(0)
    
    bars = ax.bar(range(len(MG_VALUES)), conc_accuracy, color='coral', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mg Concentration (mM)', fontsize=12, fontname='Arial')
    ax.set_ylabel('Accuracy', fontsize=12, fontname='Arial')
    ax.set_title('Accuracy by Concentration', fontsize=14, fontweight='bold', fontname='Arial')
    ax.set_xticks(range(len(MG_VALUES)))
    ax.set_xticklabels(MG_VALUES, rotation=45, fontname='Arial')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc_conc in zip(bars, conc_accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc_conc:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold', fontname='Arial')
    plt.show()
    input("\nPress Enter to continue...")


def visualize_pca(X_train_scaled, y_train_encoded):
    fig, ax = plt.subplots(figsize=(10, 8))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_encoded, cmap='viridis', alpha=0.7, s=15, edgecolors='black', linewidth=0.3)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, fontname='Arial')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, fontname='Arial')
    ax.set_title('PCA Distribution', fontsize=14, fontweight='bold', fontname='Arial')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mg Concentration (mM)', fontsize=10, fontname='Arial')
    cbar.set_ticks(range(len(MG_VALUES)))
    cbar.set_ticklabels(MG_VALUES)
    ax.grid(True, alpha=0.3)
    plt.show()
    input("\nPress Enter to continue...")


def visualize_scatter_prediction(y_test, y_pred_mg):
    fig, ax = plt.subplots(figsize=(10, 8))
    jitter = 0.1
    y_test_jitter = y_test + np.random.normal(0, jitter, len(y_test))
    y_pred_jitter = y_pred_mg + np.random.normal(0, jitter, len(y_pred_mg))
    
    ax.scatter(y_test_jitter, y_pred_jitter, c='blue', alpha=0.6, s=50, edgecolors='black')
    ax.plot([min(MG_VALUES)-0.5, max(MG_VALUES)+0.5], [min(MG_VALUES)-0.5, max(MG_VALUES)+0.5], 'r--', linewidth=2, label='Ideal')
    ax.set_xlabel('True Concentration (mM)', fontsize=12, fontname='Arial')
    ax.set_ylabel('Predicted Concentration (mM)', fontsize=12, fontname='Arial')
    ax.set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold', fontname='Arial')
    ax.legend(loc='upper left', prop={'family': 'Arial'})
    ax.grid(True, alpha=0.3)
    plt.show()
    input("\nPress Enter to continue...")


def visualize_error_distribution(y_test, y_pred_mg):
    fig, ax = plt.subplots(figsize=(10, 6))
    errors_by_conc = {}
    for mg in MG_VALUES:
        mask = y_test == mg
        if mask.sum() > 0:
            errors_by_conc[mg] = np.abs(y_test[mask] - y_pred_mg[mask])
    
    positions = range(len(errors_by_conc))
    ax.boxplot([errors_by_conc[c] for c in MG_VALUES if c in errors_by_conc], positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax.set_xlabel('Mg Concentration (mM)', fontsize=12, fontname='Arial')
    ax.set_ylabel('Absolute Error (mM)', fontsize=12, fontname='Arial')
    ax.set_title('Error Distribution by Concentration', fontsize=14, fontweight='bold', fontname='Arial')
    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in MG_VALUES if c in errors_by_conc], rotation=45, fontname='Arial')
    ax.grid(True, alpha=0.3, axis='y')
    plt.show()
    input("\nPress Enter to continue...")


def visualize_decision_boundary(X_train_scaled, y_train_encoded):
    fig, ax = plt.subplots(figsize=(10, 8))
    X_2d = X_train_scaled[:, :2]
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    knn_2d = KNeighborsClassifier(n_neighbors=3)
    knn_2d.fit(X_2d, y_train_encoded)
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train_encoded, cmap='viridis', alpha=0.7, s=15, edgecolors='black', linewidth=0.3)
    ax.set_xlabel('Feature 1 (Standardized)', fontsize=12, fontname='Arial')
    ax.set_ylabel('Feature 2 (Standardized)', fontsize=12, fontname='Arial')
    ax.set_title('KNN Decision Boundary (First 2 Features)', fontsize=14, fontweight='bold', fontname='Arial')
    ax.grid(True, alpha=0.3)
    plt.show()
    input("\nPress Enter to continue...")


def save_models(model, scaler, label_encoder):
    joblib.dump(model, f"{MODEL_SAVE_DIR}/mg_knn3_model.pkl")
    joblib.dump(scaler, f"{MODEL_SAVE_DIR}/mg_scaler.pkl")
    joblib.dump(label_encoder, f"{MODEL_SAVE_DIR}/mg_label_encoder.pkl")
    joblib.dump(MG_VALUES, f"{MODEL_SAVE_DIR}/mg_values.pkl")


if __name__ == "__main__":
    images_by_mg = load_and_crop_images()
    X_train, y_train, X_test, y_test = build_dataset_with_replacement(images_by_mg)
    model, scaler, label_encoder, y_pred, y_test_encoded, X_train_scaled, y_train_encoded, acc, mae, y_pred_mg = train_knn_model(X_train, y_train, X_test, y_test)
    visualize_confusion_matrix(y_test_encoded, y_pred)
    visualize_accuracy_by_concentration(y_test_encoded, y_pred)
    visualize_pca(X_train_scaled, y_train_encoded)
    visualize_scatter_prediction(y_test, y_pred_mg)
    visualize_error_distribution(y_test, y_pred_mg)
    visualize_decision_boundary(X_train_scaled, y_train_encoded)
    save_models(model, scaler, label_encoder)