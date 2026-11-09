import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.svm import SVC
import seaborn as sns
import joblib
import random

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
random.seed(42)

IMAGE_DIR = r"D:/lingshi/CL/cl1"
FEATURES_DIR = r"D:/lingshi/cl_features"
MODEL_SAVE_DIR = r"D:/lingshi"

ROI_X_START, ROI_Y_START = 1441, 2039
ROI_X_END, ROI_Y_END = 1604, 2209

ROI_WIDTH = ROI_X_END - ROI_X_START
ROI_HEIGHT = ROI_Y_END - ROI_Y_START

GRID_SIZE = 3

TRAIN_NOISE_SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5]

TEST_NOISE_SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

IMAGES_PER_CONC = 10
TRAIN_PER_CONC = 7
TEST_PER_CONC = 3

CL_VALUES = [0, 5, 15, 30, 45, 60, 75, 90, 100]

SVM_C = 10
SVM_GAMMA = 'scale'
SVM_KERNEL = 'rbf'

Path(FEATURES_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

def load_and_crop_images():
    folder_path = Path(IMAGE_DIR)
    images_by_cl = defaultdict(list)
    
    if not folder_path.exists():
        print(f"错误: 目录不存在 {IMAGE_DIR}")
        return images_by_cl
    
    for img_path in folder_path.glob("cl=*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        filename = img_path.stem
        if '_' not in filename:
            continue
            
        cl_part = filename.split('_')[0].replace('cl=', '')
        num_part = int(filename.split('_')[1])
        
        try:
            cl_value = int(cl_part)
        except ValueError:
            continue
        
        if cl_value not in CL_VALUES:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi = img_rgb[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
        
        if roi.size == 0:
            continue
        
        images_by_cl[cl_value].append({
            'roi': roi,
            'num': num_part,
            'filename': img_path.name
        })
    
    for cl in images_by_cl:
        images_by_cl[cl].sort(key=lambda x: x['num'])
    
    return images_by_cl

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

def create_abnormal_roi():
    abnormal_roi = np.random.randint(0, 256, (ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    return abnormal_roi

def build_dataset(images_by_cl):
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for cl_value in CL_VALUES:
        if cl_value not in images_by_cl:
            print(f"警告: 浓度 {cl_value} 没有图片")
            continue
        
        images = images_by_cl[cl_value]
        train_images = images[:TRAIN_PER_CONC]
        test_images = images[TRAIN_PER_CONC:TRAIN_PER_CONC + TEST_PER_CONC]
        
        print(f"浓度 {cl_value}: 训练 {len(train_images)} 张, 测试 {len(test_images)} 张")
        
        for img_info in train_images:
            roi = img_info['roi']
            X_train.append(extract_grid_features(roi))
            y_train.append(cl_value)
            for sigma in TRAIN_NOISE_SIGMAS:
                for _ in range(4):
                    roi_noisy = add_gaussian_noise(roi, sigma)
                    X_train.append(extract_grid_features(roi_noisy))
                    y_train.append(cl_value)
        
        for img_info in test_images:
            roi = img_info['roi']
            X_test.append(extract_grid_features(roi))
            y_test.append(cl_value)
            for sigma in TEST_NOISE_SIGMAS:
                for _ in range(2):
                    roi_noisy = add_gaussian_noise(roi, sigma)
                    X_test.append(extract_grid_features(roi_noisy))
                    y_test.append(cl_value)
    
    print(f"测试集样本数: {len(X_test)}")
    if len(X_test) == 0:
        print("错误: 测试集为空！")
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    abnormal_indices = random.sample(range(len(X_test)), 2)
    abnormal_roi = create_abnormal_roi()
    abnormal_features = extract_grid_features(abnormal_roi)
    for idx in abnormal_indices:
        X_test[idx] = abnormal_features
    
    return np.array(X_train), np.array(y_train), X_test, y_test

def train_svm_model(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, random_state=42, probability=True)
    model.fit(X_train_scaled, y_train_encoded)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_encoded, y_pred)
    
    print("="*60)
    print(f"测试集总样本数: {len(y_test)}")
    print(f"正确分类数: {(y_pred == y_test_encoded).sum()}")
    print(f"错误分类数: {(y_pred != y_test_encoded).sum()}")
    print(f"准确率: {acc:.4f} ({acc*100:.2f}%)")
    print("="*60)
    
    print("\n分类报告:")
    print(classification_report(y_test_encoded, y_pred, target_names=[str(c) for c in CL_VALUES], zero_division=0))
    
    return model, scaler, label_encoder, y_pred, y_test_encoded, X_train_scaled, y_train_encoded, acc

def visualize_results(model, y_test, y_test_encoded, y_pred, X_train_scaled, y_train, y_train_encoded, acc):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CL_VALUES, yticklabels=CL_VALUES, annot_kws={'size': 10})
    plt.tight_layout()
    plt.show()
    input("\nPress Enter to continue...")
    
    plt.figure(figsize=(10, 6))
    conc_accuracy = []
    for i, cl in enumerate(CL_VALUES):
        mask = (y_test_encoded == i)
        if mask.sum() > 0:
            acc_conc = (y_pred[mask] == i).mean()
            conc_accuracy.append(acc_conc)
        else:
            conc_accuracy.append(0)
    plt.bar(range(len(CL_VALUES)), conc_accuracy, color='coral', edgecolor='black', linewidth=0.5)
    plt.xticks(range(len(CL_VALUES)), CL_VALUES, rotation=45)
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    input("\nPress Enter to continue...")
    
    plt.figure(figsize=(10, 8))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.7, s=15, edgecolors='black', linewidth=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    input("\nPress Enter to continue...")
    
    plt.figure(figsize=(10, 8))
    y_pred_cl = label_encoder.inverse_transform(y_pred)
    jitter = 0.3
    y_test_jitter = y_test + np.random.normal(0, jitter, len(y_test))
    y_pred_jitter = y_pred_cl + np.random.normal(0, jitter, len(y_pred_cl))
    plt.scatter(y_test_jitter, y_pred_jitter, c='blue', alpha=0.6, s=50, edgecolors='black')
    plt.plot([min(CL_VALUES)-5, max(CL_VALUES)+5], [min(CL_VALUES)-5, max(CL_VALUES)+5], 'r--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    input("\nPress Enter to continue...")
    
    plt.figure(figsize=(10, 8))
    feature_corr = np.corrcoef(X_train_scaled.T)
    corr_subset = feature_corr[:27, :27]
    plt.imshow(corr_subset, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.tight_layout()
    plt.show()
    input("\nPress Enter to continue...")
    
    plt.figure(figsize=(10, 6))
    support_per_class = model.n_support_
    classes = [str(c) for c in CL_VALUES]
    plt.bar(classes, support_per_class, color='steelblue', edgecolor='black', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    input("\nPress Enter to continue...")

def save_models(model, scaler, label_encoder):
    joblib.dump(model, f"{MODEL_SAVE_DIR}/cl_svm_model.pkl")
    joblib.dump(scaler, f"{MODEL_SAVE_DIR}/cl_scaler.pkl")
    joblib.dump(label_encoder, f"{MODEL_SAVE_DIR}/cl_label_encoder.pkl")

if __name__ == "__main__":
    images_by_cl = load_and_crop_images()
    X_train, y_train, X_test, y_test = build_dataset(images_by_cl)
    model, scaler, label_encoder, y_pred, y_test_encoded, X_train_scaled, y_train_encoded, acc = train_svm_model(X_train, y_train, X_test, y_test)
    visualize_results(model, y_test, y_test_encoded, y_pred, X_train_scaled, y_train, y_train_encoded, acc)
    save_models(model, scaler, label_encoder)