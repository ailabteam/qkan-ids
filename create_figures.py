import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from dataset import IntrusionDataset, get_feature_dim
from train_qkan_v2 import QKANAutoencoder # Sử dụng định nghĩa từ V2

# --- Cấu hình ---
PROCESSED_DIR = Path("./processed_data/")
MODEL_SAVE_DIR = Path("./models/")
FIGURES_DIR = Path("./figures/")
FIGURES_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DPI = 600

# Cài đặt style cho plot
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def load_model(model_class, model_filename):
    """Hàm tiện ích để tải một mô hình."""
    input_dim = get_feature_dim()
    model = model_class(input_dim, [64, 32], 16)
    model_path = MODEL_SAVE_DIR / model_filename
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return None
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_reconstruction_errors(model, dataloader):
    """Tính toán sai số tái tạo."""
    criterion = nn.MSELoss(reduction='none')
    all_errors, all_labels = [], []
    with torch.no_grad():
        for inputs, _, labels in tqdm(dataloader, desc=f"Getting errors for {model.__class__.__name__}"):
            inputs = inputs.to(DEVICE)
            reconstructions = model(inputs)
            errors = criterion(reconstructions, inputs).mean(dim=1)
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_errors), np.concatenate(all_labels)

def plot_figure1_convergence():
    """Vẽ và lưu Hình 1: Biểu đồ hội tụ loss."""
    print("--- Generating Figure 1: Convergence Plot ---")
    try:
        mlp_history = joblib.load(MODEL_SAVE_DIR / "MLPAutoencoder_history.joblib")
        qkan_history = joblib.load(MODEL_SAVE_DIR / "QKANAutoencoder_history.joblib")
    except FileNotFoundError as e:
        print(f"Error: Could not find history file. {e}")
        print("Please run create_history_files.py first.")
        return

    epochs = range(1, len(mlp_history) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mlp_history, 'o-', label='MLP-AE')
    plt.plot(epochs, qkan_history, 's-', label='QKAN-AE')
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Loss (MSE)", fontsize=12)
    plt.title("Validation Loss Convergence", fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.xticks(epochs)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / "figure1_convergence.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved Figure 1 to {save_path}")
    plt.close()

def plot_figure2_error_distribution(model):
    """Vẽ và lưu Hình 2: Phân phối sai số."""
    print("\n--- Generating Figure 2: Error Distribution Plot ---")
    test_dataset = IntrusionDataset(PROCESSED_DIR, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    errors, labels = get_reconstruction_errors(model, test_loader)
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(errors[labels==0], label='Benign', fill=True, lw=2.5)
    sns.kdeplot(errors[labels==1], label='Attack', fill=True, lw=2.5)
    
    plt.xlabel("Reconstruction Error (MSE)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Distribution of Reconstruction Errors (QKAN-AE)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()

    save_path = FIGURES_DIR / "figure2_error_distribution.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved Figure 2 to {save_path}")
    plt.close()

def plot_figure3_interpretability(model):
    """Vẽ và lưu Hình 3: Case study về tính diễn giải."""
    print("\n--- Generating Figure 3: Interpretability Plot ---")
    
    try:
        columns = joblib.load(PROCESSED_DIR / 'columns.pkl')
    except FileNotFoundError:
        print("Error: columns.pkl not found.")
        return

    # Đặc trưng này thường quan trọng trong các cuộc tấn công Brute-force
    feature_to_plot_name = 'Flow IAT Max' 
    if feature_to_plot_name not in columns:
        print(f"Warning: '{feature_to_plot_name}' not in columns. Using first feature instead.")
        feature_to_plot_index = 0
        feature_to_plot_name = columns[0]
    else:
        feature_to_plot_index = columns.index(feature_to_plot_name)
    
    print(f"Plotting activation function for feature: '{feature_to_plot_name}' (index {feature_to_plot_index})")

    fig, ax = plt.subplots(figsize=(8, 5))
    
    if isinstance(model, QKANAutoencoder):
        encoder = model.encoder
        # plot() của qkan cần một axis để vẽ
        encoder.plot(in_vars=[feature_to_plot_index], out_vars=[0], ax=ax, plot_beta=False)
        ax.set_title(f"Learned Activation for '{feature_to_plot_name}'\n(QKAN-AE)", fontsize=14, fontweight='bold')
    else:
        print("Interpretability plot is only supported for QKAN models.")
        return
        
    ax.set_xlabel(f"Input Value (Scaled)", fontsize=12)
    ax.set_ylabel("Activation Output", fontsize=12)
    fig.tight_layout()

    save_path = FIGURES_DIR / "figure3_interpretability.png"
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved Figure 3 to {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate figures for the paper.")
    parser.add_argument('figures', type=str, nargs='+', choices=['all', '1', '2', '3'], 
                        help="Which figures to generate ('all', '1', '2', '3').")
    args = parser.parse_args()

    # Tải mô hình QKAN-AE v2 (tốt nhất) để sử dụng cho Hình 2 và 3
    qkan_model = load_model(QKANAutoencoder, "best_qkan_autoencoder_v2_15epochs.pth")
    
    if 'all' in args.figures or '1' in args.figures:
        plot_figure1_convergence()
    
    if qkan_model:
        if 'all' in args.figures or '2' in args.figures:
            plot_figure2_error_distribution(qkan_model)
        
        if 'all' in args.figures or '3' in args.figures:
            plot_figure3_interpretability(qkan_model)
    else:
        print("Skipping Figure 2 and 3 because QKAN model could not be loaded.")
