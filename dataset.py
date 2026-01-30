import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

class IntrusionDataset(Dataset):
    """
    Dataset tối ưu hóa đọc file Parquet cho UNSW-NB15 và CIC-IDS2018
    """
    def __init__(self, data_path: Path):
        self.data_path = data_path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file tại {data_path}")

        # Đọc Parquet cực nhanh và tốn ít RAM hơn CSV
        self.df = pd.read_parquet(data_path)
        
        # Tách features và label (Trong file Parquet của bạn, cột nhãn tên là 'label')
        self.features = self.df.drop(columns=['label']).values
        self.labels = self.df['label'].values
        
        # Chuyển sang tensor một lần để tiết kiệm thời gian khi training (do GPU 4090 rất nhanh)
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Đối với Autoencoder: input và target đều là x
        x = self.features[idx]
        label = self.labels[idx]
        return x, x, label 

def get_feature_dim(data_path: Path):
    """Lấy số chiều feature từ file parquet"""
    df_sample = pd.read_parquet(data_path, engine='pyarrow').iloc[:1]
    return df_sample.drop(columns=['label']).shape[1]
