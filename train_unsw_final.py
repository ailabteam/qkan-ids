import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import joblib
import time

from qkan import QKAN
from dataset import IntrusionDataset, get_feature_dim

# --- CẤU HÌNH SIÊU THAM SỐ (Khớp với Paper) ---
TRAIN_DATA_PATH = Path("./processed_data_unsw/train.parquet")
MODEL_SAVE_DIR = Path("./models/")
MODEL_SAVE_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048 # Tăng lên để tận dụng 4090
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
ENCODING_DIMS = [64, 32]
BOTTLENECK_DIM = 16

# --- ĐỊNH NGHĨA AUTOENCODER ---
class QKANAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        encoder_layers = [input_dim] + ENCODING_DIMS + [BOTTLENECK_DIM]
        decoder_layers = encoder_layers[::-1]
        
        # num_qlayers=1 theo code cũ của bạn
        self.encoder = QKAN(encoder_layers, num_qlayers=1, device=DEVICE)
        self.decoder = QKAN(decoder_layers, num_qlayers=1, device=DEVICE)

    def forward(self, x):
        return self.decoder(self.encoder(x))

def main():
    print(f"--- STARTING FINAL TRAINING (UNSW-NB15) ---")
    
    # 1. Chuẩn bị dữ liệu
    input_dim = get_feature_dim(TRAIN_DATA_PATH)
    train_ds = IntrusionDataset(TRAIN_DATA_PATH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. Khởi tạo mô hình
    model = QKANAutoencoder(input_dim).to(DEVICE)
    
    # Tối ưu hóa cho 2 card RTX 4090
    if torch.cuda.device_count() > 1:
        print(f"Sử dụng {torch.cuda.device_count()} GPUs (DataParallel)!")
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. Vòng lặp huấn luyện
    history = []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, _, _ in pbar:
            inputs = inputs.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1} hoàn thành. Average Loss: {avg_loss:.6f}")

    # 4. Lưu kết quả
    total_time = time.time() - start_time
    print(f"\nHuấn luyện xong trong {total_time:.2f} giây.")
    
    # Lưu Model (Lưu weights gốc để tránh lỗi DataParallel khi load)
    final_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(final_model_state, MODEL_SAVE_DIR / "best_qkan_unsw_final.pth")
    
    # Lưu history để vẽ biểu đồ hội tụ (Figure 1 của Revision)
    joblib.dump(history, MODEL_SAVE_DIR / "qkan_unsw_history_final.joblib")
    print(f"Model và History đã được lưu vào {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    main()
