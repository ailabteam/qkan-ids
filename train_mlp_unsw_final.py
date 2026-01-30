import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import joblib
import time

from dataset import IntrusionDataset, get_feature_dim

# --- CẤU HÌNH (Y hệt QKAN để so sánh công bằng) ---
TRAIN_DATA_PATH = Path("./processed_data_unsw/train.parquet")
MODEL_SAVE_DIR = Path("./models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30 # Tăng lên 30 cho đồng bộ

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Cấu trúc đối xứng: [input -> 64 -> 32 -> 16 -> 32 -> 64 -> input]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def main():
    print(f"--- HUẤN LUYỆN MLP-AE (BASELINE) ---")
    input_dim = get_feature_dim(TRAIN_DATA_PATH)
    full_train_ds = IntrusionDataset(TRAIN_DATA_PATH)
    train_size = int(0.9 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPAutoencoder(input_dim).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for inputs, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_inputs, _, _ in val_loader:
                v_inputs = v_inputs.to(DEVICE)
                v_outputs = model(v_inputs)
                val_loss += criterion(v_outputs, v_inputs).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        print(f"MLP Epoch {epoch+1}: Train {avg_train:.6f} | Val {avg_val:.6f}")

    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
               MODEL_SAVE_DIR / "best_mlp_unsw_final.pth")
    joblib.dump(history, MODEL_SAVE_DIR / "history_mlp_unsw_final.joblib")
    print("--- MLP-AE TRAINING COMPLETE ---")

if __name__ == "__main__":
    main()
