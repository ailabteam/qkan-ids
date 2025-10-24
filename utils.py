import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from qkan import QKAN
import warnings

# Bỏ qua các cảnh báo không quan trọng khi đọc file CSV
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# --- CẤU HÌNH CHUNG CHO CÁC MÔ HÌNH ---
ENCODING_DIMS = [64, 32]
BOTTLENECK_DIM = 16
CLIP_VALUE = 5.0

# --- CÁC LỚP DATASET ---
class IntrusionDataset(Dataset):
    """Dataset cho CSE-CIC-IDS2018."""
    def __init__(self, data_dir, is_train=True):
        self.files = sorted([f for f in data_dir.glob("*.csv")])
        if not self.files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}. Please run the preprocessing script first.")
        
        df_list = [pd.read_csv(file) for file in self.files]
        self.df = pd.concat(df_list, ignore_index=True)
        self.df.dropna(inplace=True)

        if is_train:
            self.df = self.df[self.df['Label'] == 'Benign'].reset_index(drop=True)
        
        self.features = self.df.drop(columns=['Label']).values
        self.labels = self.df['Label'].apply(lambda x: 0 if x == 'Benign' else 1).values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Trả về (features, label) cho ngắn gọn
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class UNSWDataset(Dataset):
    """Dataset cho UNSW-NB15."""
    def __init__(self, data_dir, is_train=True):
        data_path = data_dir / "unsw_nb15_processed.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}. Please run the preprocessing script first.")
        
        self.df = pd.read_csv(data_path)
        self.df.dropna(inplace=True)

        if is_train:
            self.df = self.df[self.df['label'] == 0].reset_index(drop=True)
        
        self.features = self.df.drop(columns=['label']).values
        self.labels = self.df['label'].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --- CÁC LỚP MODEL ---
class QKANAutoencoder(nn.Module):
    def __init__(self, input_dim, device, use_clamp=False):
        super().__init__()
        self.use_clamp = use_clamp
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive.")
        
        encoder_layers = [input_dim] + ENCODING_DIMS + [BOTTLENECK_DIM]
        decoder_layers = encoder_layers[::-1]
        
        self.encoder = QKAN(encoder_layers, num_qlayers=1, device=device)
        self.decoder = QKAN(decoder_layers, num_qlayers=1, device=device)

    def forward(self, x):
        if self.use_clamp:
            x = torch.clamp(x, -CLIP_VALUE, CLIP_VALUE)
        return self.decoder(self.encoder(x))

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, device, use_clamp=False): # Giữ `use_clamp` cho API nhất quán
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive.")

        # Encoder
        encoder_layers = []
        in_features = input_dim
        all_dims = ENCODING_DIMS + [BOTTLENECK_DIM]
        for i, out_features in enumerate(all_dims):
            encoder_layers.append(nn.Linear(in_features, out_features))
            if i < len(all_dims) - 1:
                encoder_layers.append(nn.ReLU())
            in_features = out_features
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        dec_dims = [input_dim] + ENCODING_DIMS
        dec_dims = dec_dims[::-1]
        in_features = BOTTLENECK_DIM
        for i, out_features in enumerate(dec_dims):
            decoder_layers.append(nn.Linear(in_features, out_features))
            if i < len(dec_dims) - 1:
                decoder_layers.append(nn.ReLU())
            in_features = out_features
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # MLP không cần clamp, nhưng giữ logic này để không gây lỗi nếu cờ `use_clamp` được bật
        return self.decoder(self.encoder(x))
