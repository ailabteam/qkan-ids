import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import joblib
import time
import numpy as np

from qkan import QKAN
from dataset import IntrusionDataset, get_feature_dim

# --- CẤU HÌNH ---
TRAIN_DATA_PATH = Path("./processed_data_unsw/train.parquet")
MODEL_SAVE_DIR = Path("./models/")
MODEL_SAVE_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30

# --- MÔ HÌNH ---
class QKANAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        encoder_layers = [input_dim, 64, 32, 16]
        decoder_layers = [16, 32, 64, input_dim]
        self.encoder = QKAN(encoder_layers, num_qlayers=1, device=DEVICE)
        self.decoder = QKAN(decoder_layers, num_qlayers=1, device=DEVICE)

    def forward(self, x):
        return self.decoder(self.encoder(x))

def main():
    print(f"--- HUẤN LUYỆN QKAN-AE (UNSW-NB15) - FULL LOGGING ---")
    
    # 1. Load data và Split 90/10 để có Validation Set "sạch"
    full_train_ds = IntrusionDataset(TRAIN_DATA_PATH)
    train_size = int(0.9 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = get_feature_dim(TRAIN_DATA_PATH)
    model = QKANAutoencoder(input_dim).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 2. Logs lưu trữ
    history = {'train_loss': [], 'val_loss': []}
    
    start_train_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # --- PHASE: TRAINING ---
        model.train()
        train_epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for inputs, _, _ in pbar:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # --- PHASE: VALIDATION ---
        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for v_inputs, _, _ in val_loader:
                v_inputs = v_inputs.to(DEVICE)
                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs, v_inputs)
                val_epoch_loss += v_loss.item()

        avg_train = train_epoch_loss / len(train_loader)
        avg_val = val_epoch_loss / len(val_loader)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

    total_time = time.time() - start_train_time
    
    # 3. LƯU TRỮ TẤT CẢ LOGS
    # Lưu weights sạch
    model_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(model_to_save, MODEL_SAVE_DIR / "best_qkan_unsw_final.pth")
    
    # Lưu history (phục vụ Figure 1)
    joblib.dump(history, MODEL_SAVE_DIR / "history_unsw_final.joblib")
    
    # Lưu thông tin cấu hình (phục vụ Rebuttal Reviewer 3)
    config = {
        'input_dim': input_dim,
        'batch_size': BATCH_SIZE,
        'lr': LEARNING_RATE,
        'epochs': NUM_EPOCHS,
        'training_time_sec': total_time,
        'device_used': f"{torch.cuda.device_count()}x RTX 4090"
    }
    joblib.dump(config, MODEL_SAVE_DIR / "config_unsw_final.joblib")

    print(f"\n--- XONG! Toàn bộ Logs đã được lưu trong {MODEL_SAVE_DIR} ---")

if __name__ == "__main__":
    main()
