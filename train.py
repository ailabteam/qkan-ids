import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib
import argparse
import time
import sys

# Import tất cả các thành phần cần thiết từ file utils.py
from utils import QKANAutoencoder, MLPAutoencoder, IntrusionDataset, UNSWDataset

def set_seed(seed=42):
    """Hàm để đảm bảo kết quả có thể tái lặp."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, optimizer, criterion, epoch_num, total_epochs, device):
    """Hàm cho một epoch huấn luyện."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num}/{total_epochs} Training", leave=False)
    for inputs, _ in progress_bar: # Chỉ cần inputs cho autoencoder
        inputs = inputs.to(device)
        
        # Forward pass
        reconstructions = model(inputs)
        loss = criterion(reconstructions, inputs) # Loss được tính giữa input và output
        
        if torch.isnan(loss):
            continue
        
        # Backward pass và tối ưu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})
        
    return total_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, epoch_num, total_epochs, device):
    """Hàm cho một epoch validation."""
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num}/{total_epochs} Validating", leave=False)
    for inputs, _ in progress_bar:
        inputs = inputs.to(device)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, inputs)
        if not torch.isnan(loss):
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- HÀM MAIN ĐIỀU PHỐI ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Universal Trainer for Autoencoder Models.")
    parser.add_argument('model_type', type=str, choices=['qkan', 'mlp'], help="Type of model to train.")
    parser.add_argument('dataset', type=str, choices=['ids2018', 'unsw'], help="Dataset to use.")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs.")
    parser.add_argument('--use_clamp', action='store_true', help="Enable input clamping for the QKAN model on specific datasets.")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use.")
    args = parser.parse_args()

    # --- CẤU HÌNH ---
    set_seed(42)
    DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    MODEL_SAVE_DIR = Path("./models/")
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-4

    print("\n" + "="*50)
    print(f"Starting Training Run:")
    print(f"  Model: {args.model_type.upper()}")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Use Clamp: {args.use_clamp}")
    print(f"  Device: {DEVICE}")
    print("="*50 + "\n")

    # --- CHỌN DATASET VÀ MODEL DỰA TRÊN THAM SỐ DÒNG LỆNH ---
    if args.dataset == 'ids2018':
        DATA_DIR = Path("./processed_data/")
        DatasetClass = IntrusionDataset
        try:
            input_dim = len(joblib.load(DATA_DIR / 'columns.pkl'))
        except FileNotFoundError:
            sys.exit(f"Error: Preprocessed data for {args.dataset.upper()} not found. Please run preprocess_data.py.")
    elif args.dataset == 'unsw':
        DATA_DIR = Path("./processed_data_unsw/")
        DatasetClass = UNSWDataset
        try:
            input_dim = len(joblib.load(DATA_DIR / 'columns_unsw.pkl'))
        except FileNotFoundError:
            sys.exit(f"Error: Preprocessed data for {args.dataset.upper()} not found. Please run preprocess_unsw.py.")
    
    ModelClass = QKANAutoencoder if args.model_type == 'qkan' else MLPAutoencoder
        
    # --- CHUẨN BỊ DỮ LIỆU ---
    full_train_dataset = DatasetClass(DATA_DIR, is_train=True)
    train_size = int(0.9 * len(full_train_dataset)); val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Data loaded for {args.dataset.upper()}. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- KHỞI TẠO VÀ HUẤN LUYỆN ---
    model = ModelClass(input_dim, device=DEVICE, use_clamp=args.use_clamp).to(DEVICE)
    print(f"Model {model.__class__.__name__} initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')
    val_loss_history = []
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch + 1, args.epochs, DEVICE)
        val_loss = validate_one_epoch(model, val_loader, criterion, epoch + 1, args.epochs, DEVICE)
        val_loss_history.append(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filename = f"best_{args.model_type}_{args.dataset}.pth"
            torch.save(model.state_dict(), MODEL_SAVE_DIR / model_filename)
            print(f"  -> Val loss improved. Saved model to {MODEL_SAVE_DIR / model_filename}")
            
    end_time = time.time()
    print(f"\n--- {args.model_type.upper()} on {args.dataset.upper()} Training Complete! (Total time: {(end_time - start_time)/60:.2f} minutes) ---")
    
    # --- LƯU LẠI HISTORY ---
    history_filename = f"{args.model_type}_{args.dataset}_history.joblib"
    joblib.dump(val_loss_history, MODEL_SAVE_DIR / history_filename)
    print(f"Saved validation loss history to {MODEL_SAVE_DIR / history_filename}")
