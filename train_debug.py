import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

# --- Import các file cục bộ ---
from qkan import QKAN
from dataset import IntrusionDataset

# --- Cấu hình ---
# ... (Giữ nguyên cấu hình cũ) ...
PROCESSED_DIR = Path("./processed_data/")
MODEL_SAVE_DIR = Path("./models_debug/")
MODEL_SAVE_DIR.mkdir(exist_ok=True)
INPUT_DIM = 78
ENCODING_DIMS = [64, 32]
BOTTLECK_DIM = 16
NUM_EPOCHS = 3 # Chỉ chạy 3 epochs để debug nhanh
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_VALUE = 5.0 # Giá trị để cắt dữ liệu đầu vào

# ========== MÔ HÌNH 1: QKAN Autoencoder (Đã cải tiến) ==========
class QKANAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dims, bottleneck_dim):
        super().__init__()
        encoder_layers = [input_dim] + encoding_dims + [bottleneck_dim]
        decoder_layers = encoder_layers[::-1]
        print(f"Kiến trúc QKAN Encoder: {encoder_layers}")
        
        # Giảm num_qlayers xuống 1 để xem có ổn định hơn không
        self.encoder = QKAN(encoder_layers, num_qlayers=1, device=DEVICE)
        self.decoder = QKAN(decoder_layers, num_qlayers=1, device=DEVICE)

    def forward(self, x):
        # THÊM BƯỚC 1: Cắt giá trị đầu vào
        x = torch.clamp(x, -CLIP_VALUE, CLIP_VALUE)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ========== MÔ HÌNH 2: MLP Autoencoder (Để so sánh) ==========
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dims, bottleneck_dim):
        super().__init__()
        
        # Encoder
        encoder_layers_list = []
        in_dim = input_dim
        for out_dim in encoding_dims + [bottleneck_dim]:
            encoder_layers_list.append(nn.Linear(in_dim, out_dim))
            encoder_layers_list.append(nn.ReLU())
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers_list)
        print(f"Kiến trúc MLP Encoder được xây dựng.")

        # Decoder
        decoder_layers_list = []
        # Đảo ngược lại danh sách chiều
        dims = [input_dim] + encoding_dims + [bottleneck_dim]
        in_out_dims = list(zip(dims[::-1][:-1], dims[::-1][1:]))
        for i, (in_dim, out_dim) in enumerate(in_out_dims):
            decoder_layers_list.append(nn.Linear(in_dim, out_dim))
            if i < len(in_out_dims) - 1: # Không dùng ReLU ở lớp cuối
                decoder_layers_list.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers_list)
        print(f"Kiến trúc MLP Decoder được xây dựng.")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Vòng lặp Huấn luyện (giữ nguyên, đã có gradient clipping) ---
def train_one_epoch(model, dataloader, optimizer, criterion):
    # ... (Copy y hệt hàm train_one_epoch từ file train.py cũ đã sửa) ...
    model.train()
    total_loss = 0.0
    num_nan_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets, _ in progress_bar:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        
        reconstructions = model(inputs)
        loss = criterion(reconstructions, targets)

        if torch.isnan(loss):
            num_nan_batches += 1
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
    avg_loss = total_loss / (len(dataloader) - num_nan_batches) if (len(dataloader) - num_nan_batches) > 0 else 0
    print(f"Skipped {num_nan_batches}/{len(dataloader)} NaN batches during training.")
    return avg_loss

# --- Vòng lặp Đánh giá (giữ nguyên) ---
def validate_one_epoch(model, dataloader, criterion):
    # ... (Copy y hệt hàm validate_one_epoch từ file train.py) ...
    model.eval()
    total_loss = 0.0
    num_nan_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, targets, _ in progress_bar:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            reconstructions = model(inputs)
            loss = criterion(reconstructions, targets)
            
            if torch.isnan(loss):
                num_nan_batches += 1
                continue

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
    avg_loss = total_loss / (len(dataloader) - num_nan_batches) if (len(dataloader) - num_nan_batches) > 0 else 0
    print(f"Found {num_nan_batches}/{len(dataloader)} NaN batches during validation.")
    return avg_loss

# --- Hàm chính ---
def main():
    # ... (Copy y hệt phần đầu hàm main từ train.py) ...
    print(f"Sử dụng thiết bị: {DEVICE}")
    full_train_dataset = IntrusionDataset(PROCESSED_DIR, is_train=True)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Kích thước tập huấn luyện: {len(train_dataset)}")
    print(f"Kích thước tập validation: {len(val_dataset)}")
    
    # ========== CHỌN MÔ HÌNH ĐỂ DEBUG ==========
    # --- Lựa chọn 1: Chạy QKAN đã cải tiến ---
    # model = QKANAutoencoder(INPUT_DIM, ENCODING_DIMS, BOTTLECK_DIM).to(DEVICE)
    # print("\n--- DEBUGGING WITH IMPROVED QKAN AUTOENCODER ---")

    # --- Lựa chọn 2: Chạy MLP để so sánh ---
    model = MLPAutoencoder(INPUT_DIM, ENCODING_DIMS, BOTTLECK_DIM).to(DEVICE)
    print("\n--- DEBUGGING WITH MLP AUTOENCODER (SANITY CHECK) ---")
    # ==========================================

    if torch.cuda.device_count() > 1:
        print(f"Sử dụng {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # ... (Copy y hệt phần vòng lặp huấn luyện từ train.py) ...
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate_one_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            model_path = MODEL_SAVE_DIR / "best_debug_model.pth"
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            print(f"Validation loss cải thiện. Đã lưu mô hình tại: {model_path}")
    print("\n--- Debug run hoàn tất! ---")

if __name__ == '__main__':
    main()
