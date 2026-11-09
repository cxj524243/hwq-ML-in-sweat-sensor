import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import joblib
import random

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
random.seed(42)

IMAGE_DIR = r"D:\lingshi\glu\glu1\glu1"
FEATURES_DIR = r"D:/lingshi/glu_features"
MODEL_SAVE_DIR = r"D:/lingshi"

ROI_X_START, ROI_Y_START = 1292, 1817
ROI_X_END, ROI_Y_END = 1647, 2156

ROI_WIDTH = ROI_X_END - ROI_X_START
ROI_HEIGHT = ROI_Y_END - ROI_Y_START

GRID_SIZE = 3

TRAIN_NOISE_SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5]
TEST_NOISE_SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

IMAGES_PER_CONC = 10
TRAIN_PER_CONC = 7
TEST_PER_CONC = 3

GLU_VALUES = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]

SVM_C = 10
SVM_GAMMA = 'scale'
SVM_KERNEL = 'rbf'

Path(FEATURES_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

def load_and_crop_images():
    folder_path = Path(IMAGE_DIR)
    images_by_glu = defaultdict(list)
    
    if not folder_path.exists():
        return images_by_glu
    
    for img_path in folder_path.glob("glu=*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        filename = img_path.stem
        if '_' not in filename:
            continue
            
        glu_part = filename.split('_')[0].replace('glu=', '')
        num_part = int(filename.split('_')[1])
        
        try:
            glu_value = float(glu_part)
        except ValueError:
            continue
        
        glu_value_rounded = round(glu_value, 1)
        if glu_value_rounded not in GLU_VALUES:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi = img_rgb[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
        
        if roi.size == 0:
            continue
        
        images_by_glu[glu_value_rounded].append({
            'roi': roi,
            'num': num_part,
            'filename': img_path.name
        })
    
    for glu in images_by_glu:
        images_by_glu[glu].sort(key=lambda x: x['num'])
    
    return images_by_glu

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

def build_dataset(images_by_glu):
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for glu_value in GLU_VALUES:
        if glu_value not in images_by_glu:
            continue
        
        images = images_by_glu[glu_value]
        train_images = images[:TRAIN_PER_CONC]
        test_images = images[TRAIN_PER_CONC:TRAIN_PER_CONC + TEST_PER_CONC]
        
        for img_info in train_images:
            roi = img_info['roi']
            X_train.append(extract_grid_features(roi))
            y_train.append(glu_value)
            for sigma in TRAIN_NOISE_SIGMAS:
                for _ in range(4):
                    roi_noisy = add_gaussian_noise(roi, sigma)
                    X_train.append(extract_grid_features(roi_noisy))
                    y_train.append(glu_value)
        
        for img_info in test_images:
            roi = img_info['roi']
            X_test.append(extract_grid_features(roi))
            y_test.append(glu_value)
            for sigma in TEST_NOISE_SIGMAS:
                for _ in range(2):
                    roi_noisy = add_gaussian_noise(roi, sigma)
                    X_test.append(extract_grid_features(roi_noisy))
                    y_test.append(glu_value)
    
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
    
    print(classification_report(y_test_encoded, y_pred, target_names=[str(c) for c in GLU_VALUES], zero_division=0))
    print(f"\nOverall accuracy: {acc:.4f}")
    
    return model, scaler, label_encoder, y_pred, y_test_encoded, X_train_scaled, y_train_encoded, acc

def plot_one_pca(ax, X_pca, pc_i, pc_j, ev_ratio, y_train_encoded, colors, idx_glu):
    """画一张 PCA 子图：PC(pc_i+1) vs PC(pc_j+1)"""
    from matplotlib.patches import Ellipse

    pct_i, pct_j = ev_ratio[pc_i] * 100, ev_ratio[pc_j] * 100

    for cls_idx, glu in idx_glu:
        mask = (y_train_encoded == cls_idx)
        if mask.sum() == 0:
            continue
        x, y = X_pca[mask, pc_i], X_pca[mask, pc_j]
        ax.scatter(
            x, y,
            c=[colors[cls_idx]],
            s=24, alpha=0.7,
            edgecolor='white', linewidth=0.4,
            label=f"{glu}" if pc_i == 0 and pc_j == 1 else "",
            zorder=3,
        )
        if len(x) >= 3:
            cov = np.cov(x, y)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * 1.96 * np.sqrt(vals)
            e = Ellipse(
                xy=(np.mean(x), np.mean(y)),
                width=w, height=h, angle=theta,
                facecolor='none', edgecolor=colors[cls_idx],
                lw=1.0, ls='--', alpha=0.7, zorder=2,
            )
            ax.add_patch(e)

    ax.set_xlabel(f'PC{pc_i+1}  ({pct_i:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC{pc_j+1}  ({pct_j:.1f}%)', fontsize=10)
    ax.axhline(0, color='gray', lw=0.4, ls=':', alpha=0.4, zorder=1)
    ax.axvline(0, color='gray', lw=0.4, ls=':', alpha=0.4, zorder=1)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.4)

def plot_pca_scatter(X_train_scaled, y_train_encoded):
    from sklearn.decomposition import PCA

    # 改为全部特征
    n_components = X_train_scaled.shape[1]  # 27个PC
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_train_scaled)
    ev = pca.explained_variance_ratio_

    cmap = plt.get_cmap('tab10')
    n_cls = len(GLU_VALUES)
    colors = [cmap(i % 10) for i in range(n_cls)]
    idx_glu = list(enumerate(GLU_VALUES))

    # ---- 控制台输出所有 PC 方差 ----
    cum = 0
    print("\n===== Explained Variance =====")
    for i, e in enumerate(ev):
        cum += e
        print(f"  PC{i+1}: {e*100:6.2f}%  (cumulative: {cum*100:.2f}%)")
    print(f"==============================\n")
    print(f"总主成分数: {n_components}")
    print(f"总组合数: C({n_components}, 2) = {n_components * (n_components - 1) // 2}")

    # ---- 全部组合：C(n, 2) ----
    all_combos = [(i, j) for i in range(n_components) for j in range(i+1, n_components)]
    n_combos = len(all_combos)
    
    # 每页显示 6 个（2行×3列）
    combos_per_page = 6
    n_pages = (n_combos + combos_per_page - 1) // combos_per_page
    
    print(f"共 {n_pages} 页，每页 {combos_per_page} 个组合")
    
    for page in range(n_pages):
        start_idx = page * combos_per_page
        end_idx = min(start_idx + combos_per_page, n_combos)
        page_combos = all_combos[start_idx:end_idx]
        
        n_rows = 2
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes_flat = axes.flat
        
        for ax, (pi, pj) in zip(axes_flat, page_combos):
            plot_one_pca(ax, X_pca, pi, pj, ev, y_train_encoded, colors, idx_glu)
        
        # 隐藏多余的子图
        for ax in axes_flat[len(page_combos):]:
            ax.set_visible(False)
        
        # 图例（只在第一页显示）
        if page == 0:
            handles, labels = [], []
            for cls_idx, glu in idx_glu:
                mask = (y_train_encoded == cls_idx)
                if mask.sum() == 0:
                    continue
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=colors[cls_idx],
                                          markersize=10, label=f'{glu} mM'))
            fig.legend(handles, labels, loc='lower center', ncol=4,
                       fontsize=10, frameon=True, title='Concentration',
                       title_fontsize=11)
        
        fig.suptitle(f'PCA: All PC Combinations (Page {page+1}/{n_pages})',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        save_path = f"{MODEL_SAVE_DIR}/pca_all_combos_page{page+1}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[SAVED] Page {page+1}/{n_pages} -> {save_path}")
        plt.show()
        
        # 每页之间暂停一下，防止内存爆炸
        if page < n_pages - 1:
            input(f"按 Enter 继续显示第 {page+2} 页...")
            


if __name__ == "__main__":
    images_by_glu = load_and_crop_images()
    X_train, y_train, X_test, y_test = build_dataset(images_by_glu)
    model, scaler, label_encoder, y_pred, y_test_encoded, X_train_scaled, y_train_encoded, acc = train_svm_model(X_train, y_train, X_test, y_test)
    plot_pca_scatter(X_train_scaled, y_train_encoded)
    