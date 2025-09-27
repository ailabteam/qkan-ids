import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib
import os

from qkan import QKAN
from dataset import IntrusionDataset, get_feature_dim

# --- Cấu hình ---
def set_seed(seed=42):
    # ... (giống hệt code cũ)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

PROCESSED_DIR = Path("./processed_data/")
MODEL_SAVE_DIR = Path("./models/")
MODEL_SAVE_DIR.mkdir(exist_ok=True)

# Hyperparameters
INPUT_DIM = get_feature_dim()
ENCODING_DIMS = [64, 32]
BOTTLECK_DIM = 16
NUM_EPOCHS = 15
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_VALUE = 5.0

# --- Lấy lại các lớp và hàm từ train_qkan_v2 ---
# (Để đơn giản, chúng ta copy lại chúng ở đây)
class QKANAutoencoder(nn.Module):
    # ... (copy y hệt lớp QKANAutoencoder từ train_qkan_v2.py)
    def __init__(self, input_dim, encoding_dims, bottleneck_dim):
        super().__init__()
        if input_dim <= 0: raise ValueError("Input dimension must be positive.")
        encoder_layers = [input_dim] + encoding_dims + [bottleneck_dim]
        decoder_layers = encoder_layers[::-1]
        print(f"QKAN Encoder Architecture: {encoder_layers}")
        print(f"QKAN Decoder Architecture: {decoder_layers}")
        self.encoder = QKAN(encoder_layers, num_qlayers=1, device=DEVICE)
        self.decoder = QKAN(decoder_layers, num_qlayers=1, device=DEVICE)
    def forward(self, x):
        x = torch.clamp(x, -CLIP_VALUE, CLIP_VALUE)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_one_epoch(model, dataloader, optimizer, criterion):
    # ... (copy y hệt hàm train_one_epoch)
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
    for inputs, targets, _ in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, targets)
        if torch.isnan(loss): continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})
    return total_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion):
    # ... (copy y hệt hàm validate_one_epoch)
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Validating", leave=False)
    for inputs, targets, _ in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, targets)
        if not torch.isnan(loss): total_loss += loss.item()
    return total_loss / len(dataloader)


# --- Hàm chính ---
def main():
    if INPUT_DIM <= 0: return

    print(f"--- Resuming Training QKAN Autoencoder V2 (15 Epochs) ---")
    print(f"Using device: {DEVICE}")
    
    # ... (Tải dữ liệu giống hệt)
    full_train_dataset = IntrusionDataset(PROCESSED_DIR, is_train=True)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    model = QKANAutoencoder(INPUT_DIM, ENCODING_DIMS, BOTTLECK_DIM).to(DEVICE)
    
    # === LOGIC KHÔI PHỤC ===
    model_path = MODEL_SAVE_DIR / "best_qkan_autoencoder_v2_15epochs.pth"
    if model_path.exists():
        print(f"Found existing model at {model_path}. Loading weights to resume training.")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("No existing model found. Starting training from scratch.")
    # ========================

    criterion = nn.MSELoss()
    # Giảm learning rate một chút để tinh chỉnh (fine-tuning)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE / 5) 

    global epoch
    # Lấy val_loss cuối cùng làm điểm bắt đầu để so sánh
    try:
        # Giả sử val_loss cuối cùng là từ epoch 14
        best_val_loss = 0.007803 
    except:
        best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate_one_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> Val loss improved. Saved model to {model_path}")

    print("\n--- Resumed training complete! ---")

if __name__ == '__main__':
    main()
