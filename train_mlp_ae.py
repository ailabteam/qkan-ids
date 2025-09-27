import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib

from dataset import IntrusionDataset, get_feature_dim

# --- Cấu hình ---
def set_seed(seed=42):
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

# Hyperparameters (giữ tương tự như QKAN để so sánh công bằng)
INPUT_DIM = get_feature_dim()
ENCODING_DIMS = [64, 32]
BOTTLECK_DIM = 16
NUM_EPOCHS = 15 
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Mô hình MLP Autoencoder ---
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dims, bottleneck_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive.")
        
        # --- Encoder ---
        encoder_layers = []
        in_features = input_dim
        all_dims = encoding_dims + [bottleneck_dim]
        for i, out_features in enumerate(all_dims):
            encoder_layers.append(nn.Linear(in_features, out_features))
            # Không thêm ReLU ở lớp cuối cùng của encoder
            if i < len(all_dims) - 1:
                encoder_layers.append(nn.ReLU())
            in_features = out_features
        self.encoder = nn.Sequential(*encoder_layers)
        print(f"MLP Encoder built: {self.encoder}")

        # --- Decoder ---
        decoder_layers = []
        # Đảo ngược kiến trúc
        dec_dims = [input_dim] + encoding_dims
        dec_dims = dec_dims[::-1]
        in_features = bottleneck_dim
        for i, out_features in enumerate(dec_dims):
            decoder_layers.append(nn.Linear(in_features, out_features))
            # Không thêm ReLU ở lớp cuối cùng của decoder
            if i < len(dec_dims) - 1:
                decoder_layers.append(nn.ReLU())
            in_features = out_features
        self.decoder = nn.Sequential(*decoder_layers)
        print(f"MLP Decoder built: {self.decoder}")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Các hàm train_one_epoch và validate_one_epoch ---
# (Đây là các hàm tiêu chuẩn, copy y hệt từ các file train trước)
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
    for inputs, targets, _ in progress_bar:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, targets)
        if torch.isnan(loss): continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})
    return total_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Validating", leave=False)
    for inputs, targets, _ in progress_bar:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, targets)
        if not torch.isnan(loss): total_loss += loss.item()
    return total_loss / len(dataloader)


# --- Hàm chính ---
def main():
    if INPUT_DIM <= 0: return

    print(f"--- Training MLP Autoencoder Baseline ---")
    print(f"Using device: {DEVICE}")
    print(f"Input feature dimension: {INPUT_DIM}")
    
    full_train_dataset = IntrusionDataset(PROCESSED_DIR, is_train=True)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    model = MLPAutoencoder(INPUT_DIM, ENCODING_DIMS, BOTTLECK_DIM).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    global epoch
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate_one_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = MODEL_SAVE_DIR / "best_mlp_autoencoder.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  -> Val loss improved. Saved model to {model_path}")

    print("\n--- MLP-AE Training complete! ---")

if __name__ == '__main__':
    main()
