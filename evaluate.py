import torch
import torch.multiprocessing
# Thêm 2 dòng sau để giải quyết lỗi "Too many open files"
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from tqdm import tqdm
import joblib
import argparse

from dataset import IntrusionDataset, get_feature_dim
# Import các kiến trúc mô hình cần thiết
from train import QKANAutoencoder # Dùng chung định nghĩa QKAN-AE
from train_mlp_ae import MLPAutoencoder 

# --- Cấu hình ---
PROCESSED_DIR = Path("./processed_data/")
MODEL_SAVE_DIR = Path("./models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048 # Sử dụng batch size lớn cho việc đánh giá để tăng tốc

def evaluate_model(model_type: str):
    """
    Hàm chính để đánh giá một mô hình Autoencoder.
    :param model_type: 'qkan', 'qkan_v2', hoặc 'mlp'
    """
    print(f"--- Evaluating {model_type.upper()} Autoencoder ---")
    
    # 1. Tải dữ liệu kiểm thử
    print("Loading test dataset (Benign and Attack samples)...")
    test_dataset = IntrusionDataset(PROCESSED_DIR, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Tải mô hình đã huấn luyện
    input_dim = get_feature_dim()
    if input_dim <= 0: return

    # Chọn lớp mô hình và đường dẫn file dựa trên model_type
    if model_type == 'qkan':
        model = QKANAutoencoder(input_dim, [64, 32], 16)
        model_path = MODEL_SAVE_DIR / "best_qkan_autoencoder.pth"
    elif model_type == 'qkan_v2':
        model = QKANAutoencoder(input_dim, [64, 32], 16)
        model_path = MODEL_SAVE_DIR / "best_qkan_autoencoder_v2_15epochs.pth"
    elif model_type == 'mlp':
        model = MLPAutoencoder(input_dim, [64, 32], 16)
        model_path = MODEL_SAVE_DIR / "best_mlp_autoencoder.pth"
    else:
        raise ValueError("Invalid model_type. Choose from 'qkan', 'qkan_v2', 'mlp'.")

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model weights from {model_path}...")
    # Tải trọng số vào mô hình.
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    model.to(DEVICE)
    model.eval()

    # 3. Tính toán sai số tái tạo cho toàn bộ dữ liệu
    print("Calculating reconstruction errors...")
    criterion = nn.MSELoss(reduction='none') 
    all_errors = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            reconstructions = model(inputs)
            errors = criterion(reconstructions, targets).mean(dim=1)
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels) # 0: Benign, 1: Attack

    # 4. Tìm ngưỡng (threshold) tốt nhất
    print("Finding the best threshold for F1-score on the Attack class...")
    # Chuyển đổi all_labels để lớp Attack là lớp positive (1)
    y_true_for_pr = (all_labels == 1).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_for_pr, all_errors)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    best_threshold_idx = np.argmax(f1_scores[:-1]) # Loại bỏ giá trị cuối cùng có thể không ổn định
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    print(f"Best threshold = {best_threshold:.6f} (achieves best Attack F1-score = {best_f1:.4f})")

    # 5. Phân loại và in kết quả
    y_pred = (all_errors > best_threshold).astype(int)

    print("\n" + "="*50)
    print("              Classification Report")
    print("="*50)
    # target_names: 0 là 'Benign', 1 là 'Attack'
    print(classification_report(all_labels, y_pred, target_names=['Benign', 'Attack']))
    
    # Tính AUC, sử dụng error scores trực tiếp
    auc_score = roc_auc_score(all_labels, all_errors)
    print(f"AUC Score: {auc_score:.4f}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Autoencoder Models for Intrusion Detection.")
    parser.add_argument('model_type', type=str, choices=['qkan', 'qkan_v2', 'mlp'], 
                        help="Type of model to evaluate ('qkan', 'qkan_v2', or 'mlp').")
    args = parser.parse_args()
    
    evaluate_model(args.model_type)
