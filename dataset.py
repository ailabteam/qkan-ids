import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import joblib

class IntrusionDataset(Dataset):
    """
    Dataset tùy chỉnh cho bộ dữ liệu IDS đã được tiền xử lý.
    """
    def __init__(self, data_dir: Path, is_train: bool):
        self.data_dir = data_dir
        self.files = sorted([f for f in self.data_dir.glob("*.csv")])
        
        print(f"Loading data for {'TRAINING' if is_train else 'TESTING'}...")
        
        df_list = [pd.read_csv(file) for file in self.files]
        self.dataframe = pd.concat(df_list, ignore_index=True)
        
        # Lớp bảo vệ cuối cùng: loại bỏ bất kỳ NaN nào có thể phát sinh
        initial_rows = len(self.dataframe)
        self.dataframe.dropna(inplace=True)
        if len(self.dataframe) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(self.dataframe)} rows with NaN.")

        if is_train:
            # Chỉ lấy dữ liệu 'Benign' cho tập huấn luyện
            self.dataframe = self.dataframe[self.dataframe['Label'] == 'Benign'].reset_index(drop=True)
        
        self.features = self.dataframe.drop(columns=['Label']).values
        self.labels = self.dataframe['Label'].apply(lambda x: 0 if x == 'Benign' else 1).values
        
        print(f"Loaded {len(self.dataframe)} samples.")
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Đối với Autoencoder, input và target là như nhau (chính là features)
        return features, features, label

def get_feature_dim():
    """Hàm tiện ích để lấy số lượng đặc trưng từ file đã lưu."""
    try:
        columns = joblib.load(Path("./processed_data/columns.pkl"))
        return len(columns)
    except FileNotFoundError:
        print("Error: 'columns.pkl' not found. Please run preprocessing first.")
        return -1

if __name__ == '__main__':
    # Test
    dim = get_feature_dim()
    print(f"Feature dimension: {dim}")
    
    if dim > 0:
        train_ds = IntrusionDataset(Path("./processed_data/"), is_train=True)
        test_ds = IntrusionDataset(Path("./processed_data/"), is_train=False)
        
        train_loader = DataLoader(train_ds, batch_size=4)
        inputs, _, labels = next(iter(train_loader))
        print("\nTrain loader check:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  All labels are Benign (0): {torch.all(labels == 0).item()}")

        test_loader = DataLoader(test_ds, batch_size=512)
        inputs, _, labels = next(iter(test_loader))
        print("\nTest loader check:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Unique labels in a batch: {torch.unique(labels).tolist()}")
