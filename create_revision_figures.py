import torch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from pathlib import Path
from train_unsw_final import QKANAutoencoder

def plot_extensive_interpretability():
    print("--- ĐANG VẼ ĐỒ THỊ GIẢI THÍCH MỞ RỘNG (REVIEWER 4) ---")
    
    # 1. Load data & model
    PROCESSED_DIR = Path("./processed_data_unsw/")
    columns = joblib.load(PROCESSED_DIR / 'columns_unsw.pkl')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = QKANAutoencoder(len(columns)).to(device)
    model.load_state_dict(torch.load("./models/best_qkan_unsw_final.pth"))
    model.eval()

    # 2. Danh sách 6 features quan trọng muốn show
    target_features = ['dur', 'sbytes', 'sttl', 'dload', 'sjit', 'swin']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    x_range = torch.linspace(-2, 2, 200).to(device) # Dải giá trị input (sau chuẩn hóa)

    for i, f_name in enumerate(target_features):
        if f_name in columns:
            idx = columns.index(f_name)
            
            # Trích xuất hàm kích hoạt từ layer đầu tiên của Encoder
            # Trong QKAN, mỗi cạnh (edge) là một hàm phi_ij
            with torch.no_grad():
                # Lấy ngẫu nhiên một hàm phi từ layer 0 ứng với feature idx
                # Giả sử chúng ta lấy phi nối tới neuron đầu tiên của layer ẩn
                # Công thức: phi(x) = cos(theta + omega1) * cos(theta * omega2 + omega3)
                alpha = model.encoder.layers[0].alpha[idx, 0]
                beta = model.encoder.layers[0].beta[idx, 0]
                omega = model.encoder.layers[0].omega[idx, 0]
                
                theta = alpha * x_range + beta
                y_range = torch.cos(theta + omega[0]) * torch.cos(theta * omega[1] + omega[2])
                
            # Vẽ đồ thị
            ax = axes[i]
            ax.plot(x_range.cpu().numpy(), y_range.cpu().numpy(), lw=2.5, color='#2c3e50')
            ax.set_title(f"Feature: {f_name.upper()}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Input (Scaled)", fontsize=10)
            ax.set_ylabel("Activation", fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            print(f"Warning: {f_name} không có trong danh sách features.")

    plt.tight_layout()
    plt.savefig("./figures/figure3_extensive_interpretability.png", dpi=300)
    print("Xong! Đã lưu đồ thị mở rộng vào thư mục figures.")

if __name__ == "__main__":
    plot_extensive_interpretability()
