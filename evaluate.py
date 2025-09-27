import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
import joblib
import argparse

from dataset import IntrusionDataset, get_feature_dim
# Import cả hai kiến trúc mô hình
from train import QKANAutoencoder
from train_kan_ae import KANAutoencoder
# (Sau này có thể thêm MLPAutoencoder nếu cần)

# --- Cấu hình ---
PROCESSED_DIR = Path("./processed_data/")
MODEL_SAVE_DIR = Path("./models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048 # Sử dụng batch size lớn cho việc đánh giá để tăng tốc

def evaluate_model(model_type: str):
    """
    Hàm chính để đánh giá một mô hình Autoencoder.
    :param model_type: 'qkan' hoặc 'kan'
    """
    print(f"--- Evaluating {model_type.upper()} Autoencoder ---")
    
    # 1. Tải dữ liệu kiểm thử
    print("Loading test dataset (Benign and Attack samples)...")
    # is_train=False để tải toàn bộ dữ liệu
    test_dataset = IntrusionDataset(PROCESSED_DIR, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Tải mô hình đã huấn luyện
    input_dim = get_feature_dim()
    if input_dim <= 0: return

    # Chọn lớp mô hình và đường dẫn file dựa trên model_type
    if model_type == 'qkan':
        model = QKANAutoencoder(input_dim, [64, 32], 16)
        model_path = MODEL_SAVE_DIR / "best_qkan_autoencoder.pth"
    elif model_type == 'kan':
        model = KANAutoencoder(input_dim, [64, 32], 16)
        model_path = MODEL_SAVE_DIR / "best_kan_autoencoder.pth"
    else:
        raise ValueError("Invalid model_type. Choose 'qkan' or 'kan'.")

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model weights from {model_path}...")
    # Tải trọng số vào mô hình. Cần xử lý trường hợp DataParallel
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except RuntimeError:
        # Nếu model được lưu từ nn.DataParallel, nó sẽ có prefix 'module.'
        print("Model was likely saved with DataParallel. Adjusting keys...")
        state_dict = torch.load(model_path, map_location=DEVICE)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.to(DEVICE)
    model.eval()

    # 3. Tính toán sai số tái tạo cho toàn bộ dữ liệu
    print("Calculating reconstruction errors...")
    criterion = nn.MSELoss(reduction='none') # Tính MSE cho từng sample, không lấy trung bình
    all_errors = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            reconstructions = model(inputs)
            # Tính MSE trên từng sample trong batch
            errors = criterion(reconstructions, targets).mean(dim=1)
            
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels) # 0: Benign, 1: Attack

    # 4. Tìm ngưỡng (threshold) tốt nhất
    print("Finding the best threshold for F1-score...")
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors)
    # Tính F1 score cho từng ngưỡng, tránh chia cho 0
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    print(f"Best threshold = {best_threshold:.6f} (achieves F1-score = {best_f1:.4f})")

    # 5. Phân loại và in kết quả
    # y_pred: 1 (Attack) nếu error > threshold, 0 (Benign) nếu không
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
    # Sử dụng argparse để chọn model cần đánh giá từ command line
    parser = argparse.ArgumentParser(description="Evaluate Autoencoder Models for Intrusion Detection.")
    parser.add_argument('model_type', type=str, choices=['qkan', 'kan'], help="Type of model to evaluate ('qkan' or 'kan').")
    args = parser.parse_args()
    
    evaluate_model(args.model_type)
