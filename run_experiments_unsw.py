import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib
import argparse
import time
import warnings

# --- Bỏ qua các cảnh báo không quan trọng ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import các lớp mô hình từ các file đã có
from qkan import QKAN
from train_mlp_ae import MLPAutoencoder

# --- CẤU HÌNH THÍ NGHIỆM ---
DATA_DIR_UNSW = Path("./processed_data_unsw/")
MODEL_SAVE_DIR = Path("./models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 15
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

# --- LỚP DATASET MỚI CHO UNSW-NB15 ---
class UNSWDataset(Dataset):
    def __init__(self, is_train=True):
        data_path = DATA_DIR_UNSW / "unsw_nb15_processed.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Processed UNSW-NB15 data not found at {data_path}")
        df = pd.read_csv(data_path)
        df.dropna(inplace=True)
        if is_train:
            self.dataframe = df[df['label'] == 0].reset_index(drop=True)
        else:
            self.dataframe = df
        self.features = self.dataframe.drop(columns=['label']).values
        self.labels = self.dataframe['label'].values
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, features, label

# --- CÁC HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ (Tái sử dụng) ---
def train_one_epoch(model, dataloader, optimizer, criterion, epoch_num):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} Training", leave=False)
    for inputs, targets, _ in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, targets)
        if torch.isnan(loss): continue
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})
    return total_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, epoch_num):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} Validating", leave=False)
    for inputs, targets, _ in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, targets)
        if not torch.isnan(loss): total_loss += loss.item()
    return total_loss / len(dataloader)

def train_autoencoder(model, model_name):
    print(f"\n--- Training {model_name} on UNSW-NB15 ---")
    full_train_dataset = UNSWDataset(is_train=True)
    train_size = int(0.9 * len(full_train_dataset)); val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    criterion = nn.MSELoss(); optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf'); start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch + 1)
        val_loss = validate_one_epoch(model, val_loader, criterion, epoch + 1)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_DIR / f"best_{model_name}_unsw.pth")
    end_time = time.time()
    print(f"--- {model_name} Training Complete! (Total time: {end_time - start_time:.2f}s) ---")
    return model

# --- HÀM ĐÁNH GIÁ TỔNG QUÁT ---
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
def evaluate_autoencoder(model, model_name):
    print(f"\n--- Evaluating {model_name} on UNSW-NB15 ---")
    test_dataset = UNSWDataset(is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)
    model.load_state_dict(torch.load(MODEL_SAVE_DIR / f"best_{model_name}_unsw.pth", map_location=DEVICE))
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    all_errors, all_labels = [], []
    with torch.no_grad():
        for inputs, _, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(DEVICE)
            reconstructions = model(inputs)
            errors = criterion(reconstructions, inputs).mean(dim=1)
            all_errors.append(errors.cpu().numpy()); all_labels.append(labels.cpu().numpy())
    all_errors, all_labels = np.concatenate(all_errors), np.concatenate(all_labels)
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    y_pred = (all_errors > best_threshold).astype(int)
    return {
        'F1-Score': f1_score(all_labels, y_pred), 'AUC Score': roc_auc_score(all_labels, all_errors),
        'Precision': precision_score(all_labels, y_pred), 'Recall': recall_score(all_labels, y_pred),
    }

# --- HÀM CHẠY ISOLATION FOREST ---
from sklearn.ensemble import IsolationForest
def run_isoforest():
    print("\n--- Running Isolation Forest on UNSW-NB15 ---")
    df = pd.read_csv(DATA_DIR_UNSW / "unsw_nb15_processed.csv"); df.dropna(inplace=True)
    X, y_true = df.drop(columns=['label']), df['label']
    clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    X_train_sample = X[y_true==0].sample(n=min(len(X[y_true==0]), 250000), random_state=42)
    clf.fit(X_train_sample)
    scores = -clf.decision_function(X); y_pred = (clf.predict(X) == -1).astype(int)
    return {
        'F1-Score': f1_score(y_true, y_pred), 'AUC Score': roc_auc_score(y_true, scores),
        'Precision': precision_score(y_true, y_pred), 'Recall': recall_score(y_true, y_pred),
    }
    
# --- HÀM MAIN ĐIỀU PHỐI ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full experiment suite on UNSW-NB15.")
    parser.add_argument('models', nargs='+', choices=['all', 'isoforest', 'mlp', 'qkan', 'evaluate_only'], help="Models to run or 'evaluate_only' to reprint results.")
    args = parser.parse_args()

    try:
        columns_unsw = joblib.load(DATA_DIR_UNSW / 'columns_unsw.pkl')
        INPUT_DIM = len(columns_unsw)
    except FileNotFoundError:
        print("Error: 'columns_unsw.pkl' not found. Please run preprocess_unsw.py first."); exit()

    all_results = {}

    if 'evaluate_only' in args.models:
        print("--- Re-evaluating all models on UNSW-NB15 to generate table ---")
        all_results['Isolation Forest'] = run_isoforest()
        model_mlp = MLPAutoencoder(INPUT_DIM, [64, 32], 16).to(DEVICE)
        if (MODEL_SAVE_DIR / 'best_mlp_ae_unsw.pth').exists():
            all_results['MLP-AE (15 epochs)'] = evaluate_autoencoder(model_mlp, "mlp_ae")
        class QKANAutoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dims, bottleneck_dim):
                super().__init__()
                encoder_layers = [input_dim] + encoding_dims + [bottleneck_dim]; decoder_layers = encoder_layers[::-1]
                self.encoder = QKAN(encoder_layers, num_qlayers=1, device=DEVICE)
                self.decoder = QKAN(decoder_layers, num_qlayers=1, device=DEVICE)
            def forward(self, x):
                encoded = self.encoder(x); decoded = self.decoder(encoded); return decoded
        model_qkan = QKANAutoencoder(INPUT_DIM, [64, 32], 16).to(DEVICE)
        if (MODEL_SAVE_DIR / 'best_qkan_ae_unsw.pth').exists():
             all_results['QKAN-AE (15 epochs)'] = evaluate_autoencoder(model_qkan, "qkan_ae")
    else:
        if 'isoforest' in args.models or 'all' in args.models:
            all_results['Isolation Forest'] = run_isoforest()
        if 'mlp' in args.models or 'all' in args.models:
            model_mlp = MLPAutoencoder(INPUT_DIM, [64, 32], 16).to(DEVICE)
            train_autoencoder(model_mlp, "mlp_ae")
            all_results['MLP-AE (15 epochs)'] = evaluate_autoencoder(model_mlp, "mlp_ae")
        if 'qkan' in args.models or 'all' in args.models:
            class QKANAutoencoder(nn.Module):
                def __init__(self, input_dim, encoding_dims, bottleneck_dim):
                    super().__init__()
                    encoder_layers = [input_dim] + encoding_dims + [bottleneck_dim]; decoder_layers = encoder_layers[::-1]
                    self.encoder = QKAN(encoder_layers, num_qlayers=1, device=DEVICE)
                    self.decoder = QKAN(decoder_layers, num_qlayers=1, device=DEVICE)
                def forward(self, x):
                    encoded = self.encoder(x); decoded = self.decoder(encoded); return decoded
            model_qkan = QKANAutoencoder(INPUT_DIM, [64, 32], 16).to(DEVICE)
            train_autoencoder(model_qkan, "qkan_ae")
            all_results['QKAN-AE (15 epochs)'] = evaluate_autoencoder(model_qkan, "qkan_ae")

    print("\n\n" + "="*80); print("                 FINAL RESULTS ON UNSW-NB15 DATASET"); print("="*80)
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    print(results_df.to_markdown(floatfmt=".4f"))
    print("="*80)
