import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import time
from dataset import IntrusionDataset
from train_unsw_final import QKANAutoencoder, MLPAutoencoder # Đảm bảo import đúng

def evaluate_model(model_name, model_class, data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = IntrusionDataset(data_path)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    
    # Load model
    input_dim = dataset.features.shape[1]
    model = model_class(input_dim).to(device)
    model.load_state_dict(torch.load(f"./models/best_{model_name}_unsw_final.pth"))
    model.eval()
    
    errors = []
    labels = []
    
    # Đo Inference Time (Reviewer 1 yêu cầu)
    start_time = time.time()
    with torch.no_grad():
        for inputs, _, target_labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Tính MSE từng dòng
            loss = torch.mean((outputs - inputs)**2, dim=1)
            errors.append(loss.cpu().numpy())
            labels.append(target_labels.numpy())
    
    total_inf_time = (time.time() - start_time) * 1000 # ms
    ms_per_batch = total_inf_time / len(loader)
    
    errors = np.concatenate(errors)
    labels = np.concatenate(labels)
    
    # Tìm ngưỡng tối ưu (Threshold Tuning) - Giải trình Reviewer 2 & 3
    # Chúng ta dùng Percentile 95 của dữ liệu Benign làm ngưỡng khởi đầu
    threshold = np.percentile(errors[labels == 0], 95)
    
    preds = (errors > threshold).astype(int)
    
    print(f"\n--- RESULTS FOR {model_name.upper()} ---")
    print(f"AUC: {roc_auc_score(labels, errors):.4f}")
    print(f"F1-Score: {f1_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds):.4f}")
    print(f"Recall: {recall_score(labels, preds):.4f}")
    print(f"Inference Speed: {ms_per_batch:.2f} ms/batch (Batch size 4096)")
    
    return errors, labels

if __name__ == "__main__":
    # Đánh giá QKAN
    q_err, q_lab = evaluate_model("qkan", QKANAutoencoder, "./processed_data_unsw/test.parquet")
    # Đánh giá MLP
    m_err, m_lab = evaluate_model("mlp", MLPAutoencoder, "./processed_data_unsw/test.parquet")
    
    # Lưu lại để vẽ Figure 2 (Phân phối lỗi)
    joblib.dump({'q_err': q_err, 'q_lab': q_lab, 'm_err': m_err, 'm_lab': m_lab}, 
                "./models/error_distributions_unsw.joblib")
