import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import warnings
import time
import sys

# --- Bỏ qua các cảnh báo không quan trọng ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- IMPORT CÁC LỚP MÔ HÌNH ---
try:
    from train_qkan_v2 import QKANAutoencoder
    from train_mlp_ae import MLPAutoencoder
    from sklearn.ensemble import IsolationForest
except ImportError as e:
    print(f"Error: Could not import necessary model classes. {e}")
    print("Please ensure training scripts (train_qkan_v2.py, train_mlp_ae.py) are in the same directory.")
    sys.exit(1)

# --- CẤU HÌNH ---
IDS2018_DIR = Path("./processed_data/")
UNSW_DIR = Path("./processed_data_unsw/")
MODEL_SAVE_DIR = Path("./models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ĐỊNH NGHĨA DATASET CLASSES ---
class IntrusionDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.files = sorted([f for f in data_dir.glob("*.csv")])
        if not self.files: raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
        df_list = [pd.read_csv(file) for file in self.files]
        self.dataframe = pd.concat(df_list, ignore_index=True)
        self.dataframe.dropna(inplace=True)
        
        if is_train:
            self.dataframe = self.dataframe[self.dataframe['Label'] == 'Benign'].reset_index(drop=True)
            
        self.features = self.dataframe.drop(columns=['Label']).values
        self.labels = self.dataframe['Label'].apply(lambda x: 0 if x == 'Benign' else 1).values
        
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        # Trả về (features, features, labels) để tương thích với AE
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class UNSWDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        data_path = data_dir / "unsw_nb15_processed.csv"
        if not data_path.exists(): raise FileNotFoundError(f"Data not found at {data_path}")
        
        df = pd.read_csv(data_path); df.dropna(inplace=True)
        
        if is_train:
            self.dataframe = df[df['label'] == 0].reset_index(drop=True)
        else:
            self.dataframe = df
            
        self.features = self.dataframe.drop(columns=['label']).values
        self.labels = self.dataframe['label'].values
        
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --- CÁC HÀM TIỆN ÍCH ĐÁNH GIÁ ---
def get_metrics(y_true, y_pred_binary, scores):
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    return {
        'F1-Score': f1_score(y_true, y_pred_binary, zero_division=0), 'AUC Score': roc_auc_score(y_true, scores),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0), 'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
    }

def evaluate_autoencoder(model, dataloader, model_name="Model"):
    model.eval(); criterion = nn.MSELoss(reduction='none')
    all_errors, all_labels = [], []
    with torch.no_grad():
        for inputs, _, labels in tqdm(dataloader, desc=f"Evaluating {model_name}", leave=False):
            inputs = inputs.to(DEVICE); reconstructions = model(inputs)
            errors = criterion(reconstructions, inputs).mean(dim=1)
            all_errors.append(errors.cpu().numpy()); all_labels.append(labels.cpu().numpy())
    all_errors = np.concatenate(all_errors); all_labels = np.concatenate(all_labels)
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    y_pred_binary = (all_errors > best_threshold).astype(int)
    return get_metrics(all_labels, y_pred_binary, all_errors)

def evaluate_isoforest(dataset_class, data_dir, dataset_name):
    print(f"Evaluating Isolation Forest on {dataset_name}...")
    test_df = dataset_class(data_dir, is_train=False).dataframe
    train_df = dataset_class(data_dir, is_train=True).dataframe
    label_col = 'label' if 'label' in test_df.columns else 'Label'
    y_true = test_df[label_col].apply(lambda x: 1 if (x=='Attack' or x==1) else 0)
    X_test = test_df.drop(columns=[label_col])
    X_train = train_df.drop(columns=[label_col])
    clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    clf.fit(X_train.sample(n=min(len(X_train), 250000), random_state=42))
    scores = -clf.decision_function(X_test)
    y_pred_binary = (clf.predict(X_test) == -1).astype(int)
    return get_metrics(y_true, y_pred_binary, scores)

def get_model_params_and_time(model, dataloader):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    with torch.no_grad():
        for i, (inputs, _, _) in enumerate(dataloader):
            if i >= 50: break # Chỉ đo trên 50 batch đầu
            inputs = inputs.to(DEVICE)
            starter.record()
            _ = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))
    avg_inference_time = np.mean(timings) if timings else 0
    return params, avg_inference_time

# --- HÀM TẠO BẢNG LATEX ---
def create_latex_tables(all_results):
    print("\n" + "="*80); print("                      LATEX TABLES FOR THE PAPER"); print("="*80)
    for dataset_name, results in all_results.items():
        if "Cost" in dataset_name: continue
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df[['F1-Score', 'AUC Score', 'Precision', 'Recall']]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                best_val = df[col].max()
                df[col] = df[col].apply(lambda x: f"\\textbf{{{x:.3f}}}" if x == best_val else f"{x:.3f}")
        latex_string = df.to_latex(escape=False, column_format="@{}lcccc@{}}")
        print(f"\n% --- TABLE FOR {dataset_name} ---")
        print(f"\\begin{{table}}[ht]"); print(f"\\centering")
        print(f"\\caption{{Performance on the {dataset_name.replace('_', ' ')} Dataset. Metrics are for the 'Attack' class. Best results are in \\textbf{{bold}}.}}")
        print(f"\\label{{tab:{dataset_name.lower().replace('-', '')}}}")
        print(latex_string); print(f"\\end{{table}}")

    if "Computational Cost" in all_results:
        df_cost = pd.DataFrame.from_dict(all_results["Computational Cost"], orient='index')
        df_cost['Trainable Parameters'] = df_cost['Trainable Parameters'].apply(lambda x: f"{x/1e3:.1f}K")
        df_cost['Inference Time (ms/batch)'] = df_cost['Inference Time (ms/batch)'].apply(lambda x: f"{x:.2f}")
        latex_cost = df_cost.to_latex(escape=False, column_format="@{}lrr@{}}")
        print(f"\n% --- TABLE FOR Computational Cost ---")
        print(f"\\begin{{table}}[ht]"); print(f"\\centering")
        print(f"\\caption{{Computational cost comparison on CSE-CIC-IDS2018 (Batch Size={2048}).}}")
        print(f"\\label{{tab:cost}}"); print(latex_cost); print(f"\\end{{table}}")

# --- HÀM MAIN ĐIỀU PHỐI ---
if __name__ == '__main__':
    all_final_results = {}
    print("\n" + "="*80 + "\n            COLLECTING RESULTS FOR ALL TABLES\n" + "="*80)

    # --- THÍ NGHIỆM TRÊN CSE-CIC-IDS2018 ---
    if IDS2018_DIR.exists():
        print("\n--- Evaluating on CSE-CIC-IDS2018 ---")
        results_ids2018 = {}
        input_dim_ids2018 = len(joblib.load(IDS2018_DIR / 'columns.pkl'))
        ids2018_loader = DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=2048, num_workers=4)
        
        results_ids2018['Isolation Forest'] = evaluate_isoforest(IntrusionDataset, IDS2018_DIR, "CSE-CIC-IDS2018")
        
        model_mlp_ids = MLPAutoencoder(input_dim_ids2018, [64, 32], 16).to(DEVICE)
        model_mlp_ids.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_mlp_autoencoder.pth', map_location=DEVICE))
        results_ids2018['MLP-AE (15 epochs)'] = evaluate_autoencoder(model_mlp_ids, ids2018_loader, "MLP-AE on IDS2018")

        model_qkan_5e = QKANAutoencoder(input_dim_ids2018, [64, 32], 16).to(DEVICE)
        model_qkan_5e.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_qkan_autoencoder.pth', map_location=DEVICE))
        results_ids2018['QKAN-AE (5 epochs)'] = evaluate_autoencoder(model_qkan_5e, ids2018_loader, "QKAN-AE 5e on IDS2018")
        
        model_qkan_15e = QKANAutoencoder(input_dim_ids2018, [64, 32], 16).to(DEVICE)
        model_qkan_15e.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_qkan_autoencoder_v2_15epochs.pth', map_location=DEVICE))
        results_ids2018['QKAN-AE (15 epochs)'] = evaluate_autoencoder(model_qkan_15e, ids2018_loader, "QKAN-AE 15e on IDS2018")
        
        all_final_results['CSE-CIC-IDS2018'] = results_ids2018
        
        print("\n--- Collecting Computational Cost Data ---")
        cost_data = {}
        params_mlp, time_mlp = get_model_params_and_time(model_mlp_ids, ids2018_loader)
        params_qkan, time_qkan = get_model_params_and_time(model_qkan_15e, ids2018_loader)
        cost_data['MLP-AE'] = {'Trainable Parameters': params_mlp, 'Inference Time (ms/batch)': time_mlp}
        cost_data['QKAN-AE'] = {'Trainable Parameters': params_qkan, 'Inference Time (ms/batch)': time_qkan}
        all_final_results['Computational Cost'] = cost_data

    # --- THÍ NGHIỆM TRÊN UNSW-NB15 ---
    if (UNSW_DIR / "unsw_nb15_processed.csv").exists():
        print("\n--- Evaluating on UNSW-NB15 ---")
        results_unsw = {}
        input_dim_unsw = len(joblib.load(UNSW_DIR / 'columns_unsw.pkl'))
        unsw_loader = DataLoader(UNSWDataset(UNSW_DIR, is_train=False), batch_size=2048, num_workers=4)
        
        results_unsw['Isolation Forest'] = evaluate_isoforest(UNSWDataset, UNSW_DIR, "UNSW-NB15")
        
        if (MODEL_SAVE_DIR / 'best_mlp_ae_unsw.pth').exists():
            model_mlp_unsw = MLPAutoencoder(input_dim_unsw, [64, 32], 16).to(DEVICE)
            model_mlp_unsw.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_mlp_ae_unsw.pth', map_location=DEVICE))
            results_unsw['MLP-AE (15 epochs)'] = evaluate_autoencoder(model_mlp_unsw, unsw_loader, "MLP-AE on UNSW")
        
        if (MODEL_SAVE_DIR / 'best_qkan_ae_unsw.pth').exists():
            model_qkan_unsw = QKANAutoencoder(input_dim_unsw, [64, 32], 16).to(DEVICE)
            model_qkan_unsw.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_qkan_ae_unsw.pth', map_location=DEVICE))
            results_unsw['QKAN-AE (15 epochs)'] = evaluate_autoencoder(model_qkan_unsw, unsw_loader, "QKAN-AE on UNSW")
        
        all_final_results['UNSW-NB15'] = results_unsw
    else:
        print("\nSkipping UNSW-NB15 experiments: Processed data not found.")

    # --- TẠO BẢNG ---
    create_latex_tables(all_final_results)
    
    print("\n\nAll tables generated successfully!")
