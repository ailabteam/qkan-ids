import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# --- Bỏ qua các cảnh báo không quan trọng ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- IMPORT CÁC LỚP CẦN THIẾT ---
from dataset import IntrusionDataset # For IDS2018
from run_experiments_unsw import UNSWDataset # For UNSW-NB15
from train_qkan_v2 import QKANAutoencoder
from train_mlp_ae import MLPAutoencoder
from sklearn.ensemble import IsolationForest

# --- CẤU HÌNH ---
IDS2018_DIR = Path("./processed_data/")
UNSW_DIR = Path("./processed_data_unsw/")
MODEL_SAVE_DIR = Path("./models/")
FIGURES_DIR = Path("./figures/")
FIGURES_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DPI = 600

# --- CÀI ĐẶT STYLE CHO PLOT ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# --- CÁC HÀM TIỆN ÍCH ĐÁNH GIÁ ---
def get_metrics(y_true, y_pred_binary, scores):
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    return {
        'F1-Score': f1_score(y_true, y_pred_binary, zero_division=0),
        'AUC Score': roc_auc_score(y_true, scores),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
    }

def evaluate_autoencoder(model, dataloader):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    all_errors, all_labels = [], []
    with torch.no_grad():
        for inputs, _, labels in tqdm(dataloader, desc=f"Evaluating {model.__class__.__name__}", leave=False):
            inputs = inputs.to(DEVICE)
            reconstructions = model(inputs)
            errors = criterion(reconstructions, inputs).mean(dim=1)
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels)
    
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    y_pred_binary = (all_errors > best_threshold).astype(int)
    
    return get_metrics(all_labels, y_pred_binary, all_errors)
    
def evaluate_isoforest(dataset_class):
    dataset = dataset_class(is_train=False)
    df = dataset.dataframe
    X = df.drop(columns=['label'], errors='ignore')
    if 'Label' in X.columns: X = X.drop(columns=['Label'])
    y_true = df['label'] if 'label' in df.columns else df['Label'].apply(lambda x: 1 if x=='Attack' else 0)

    clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    train_df = dataset_class(is_train=True).dataframe
    X_train = train_df.drop(columns=['label'], errors='ignore')
    if 'Label' in X_train.columns: X_train = X_train.drop(columns=['Label'])
    clf.fit(X_train.sample(n=min(len(X_train), 250000), random_state=42))

    scores = -clf.decision_function(X)
    y_pred_binary = (clf.predict(X) == -1).astype(int)
    
    return get_metrics(y_true, y_pred_binary, scores)

# --- CÁC HÀM VẼ HÌNH ---
def create_figures(qkan_model_ids2018):
    print("\n--- Generating All Figures ---")
    
    # FIGURE 1: CONVERGENCE PLOT
    try:
        mlp_history = joblib.load(MODEL_SAVE_DIR / "MLPAutoencoder_history.joblib")
        qkan_history = joblib.load(MODEL_SAVE_DIR / "QKANAutoencoder_history.joblib")
        epochs = range(1, len(mlp_history) + 1)
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, mlp_history, 'o-', label='MLP-AE')
        plt.plot(epochs, qkan_history, 's-', label='QKAN-AE')
        plt.xlabel("Epoch", fontsize=12); plt.ylabel("Validation Loss (MSE)", fontsize=12)
        plt.title("Validation Loss Convergence on CSE-CIC-IDS2018", fontsize=14, fontweight='bold')
        plt.yscale('log'); plt.xticks(epochs); plt.legend(fontsize=11); plt.tight_layout()
        save_path = FIGURES_DIR / "figure1_convergence.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved Figure 1 to {save_path}")
        plt.close()
    except FileNotFoundError as e:
        print(f"Skipping Figure 1: Could not find history file. {e}")

    # FIGURE 2: ERROR DISTRIBUTION
    test_loader_ids2018 = DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=2048)
    errors, labels = get_reconstruction_errors(qkan_model_ids2018, test_loader_ids2018)
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(errors[labels==0], label='Benign', fill=True, lw=2.5)
    sns.kdeplot(errors[labels==1], label='Attack', fill=True, lw=2.5)
    plt.xlabel("Reconstruction Error (MSE)", fontsize=12); plt.ylabel("Density", fontsize=12)
    plt.title("Distribution of Reconstruction Errors (QKAN-AE on IDS2018)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11); plt.tight_layout()
    save_path = FIGURES_DIR / "figure2_error_distribution.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved Figure 2 to {save_path}")
    plt.close()

    # FIGURE 3: INTERPRETABILITY
    try:
        columns = joblib.load(IDS2018_DIR / 'columns.pkl')
        feature_name = 'Flow IAT Max'
        feature_index = columns.index(feature_name)
        input_dim = len(columns)
        
        encoder = qkan_model_ids2018.encoder
        encoder.eval()
        
        sample_inputs, _, _ = next(iter(test_loader_ids2018))
        x_grid = torch.linspace(sample_inputs[:, feature_index].min().item(), sample_inputs[:, feature_index].max().item(), 200).to(DEVICE)
        base_input = torch.zeros(1, input_dim).to(DEVICE)
        y_grid = []
        with torch.no_grad():
            for val in x_grid:
                current_input = base_input.clone()
                current_input[0, feature_index] = val
                activation_output = encoder.layers[0](current_input)
                y_grid.append(activation_output[0, 0].item())

        plt.figure(figsize=(8, 5))
        plt.plot(x_grid.cpu().numpy(), y_grid, linewidth=2.5)
        plt.title(f"Learned Activation for '{feature_name}'\n(QKAN-AE on IDS2018)", fontsize=14, fontweight='bold')
        plt.xlabel(f"Input Value for '{feature_name}' (Scaled)", fontsize=12)
        plt.ylabel("Activation Output (Node 0, Layer 1)", fontsize=12)
        plt.grid(True); plt.tight_layout()
        save_path = FIGURES_DIR / "figure3_interpretability.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved Figure 3 to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Skipping Figure 3 due to an error: {e}")

# --- HÀM TẠO BẢNG LATEX ---
def create_latex_tables(all_results):
    print("\n\n" + "="*80)
    print("                      LATEX TABLES")
    print("="*80)
    for dataset_name, results in all_results.items():
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df[['F1-Score', 'AUC Score', 'Precision', 'Recall']]
        for col in df.columns:
            best_val = df[col].max()
            df[col] = df[col].apply(lambda x: f"\\textbf{{{x:.3f}}}" if x == best_val else f"{x:.3f}")
        latex_string = df.to_latex(escape=False, column_format="@{}lcccc@{}}")
        print(f"\n--- LaTeX Table for {dataset_name} ---")
        print(f"\\begin{{table}}[ht]")
        print(f"\\centering")
        print(f"\\caption{{Performance on the {dataset_name.replace('_', ' ')} Dataset. Metrics are for the 'Attack' class. Best results are in \\textbf{{bold}}.}}")
        print(f"\\label{{tab:{dataset_name.lower().replace('-', '')}}}")
        print(latex_string)
        print(f"\\end{{table}}")

# --- HÀM MAIN ĐIỀU PHỐI ---
if __name__ == '__main__':
    all_final_results = {}
    
    # --- THÍ NGHIỆM TRÊN CSE-CIC-IDS2018 ---
    print("\n--- Running Experiments on CSE-CIC-IDS2018 ---")
    results_ids2018 = {}
    ids2018_test_loader = DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=2048, num_workers=4)
    input_dim_ids2018 = get_feature_dim()
    
    results_ids2018['Isolation Forest'] = evaluate_isoforest(lambda is_train: IntrusionDataset(IDS2018_DIR, is_train))
    
    model_mlp_ids = MLPAutoencoder(input_dim_ids2018, [64, 32], 16).to(DEVICE)
    model_mlp_ids.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_mlp_autoencoder.pth', map_location=DEVICE))
    results_ids2018['MLP-AE (15 epochs)'] = evaluate_autoencoder(model_mlp_ids, ids2018_test_loader)

    model_qkan_ids = QKANAutoencoder(input_dim_ids2018, [64, 32], 16).to(DEVICE)
    model_qkan_ids.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_qkan_autoencoder_v2_15epochs.pth', map_location=DEVICE))
    results_ids2018['QKAN-AE (15 epochs)'] = evaluate_autoencoder(model_qkan_ids, ids2018_test_loader)
    
    all_final_results['CSE-CIC-IDS2018'] = results_ids2018

    # --- THÍ NGHIỆM TRÊN UNSW-NB15 ---
    if (UNSW_DIR / "unsw_nb15_processed.csv").exists():
        print("\n--- Running Experiments on UNSW-NB15 ---")
        results_unsw = {}
        unsw_test_loader = DataLoader(UNSWDataset(is_train=False), batch_size=2048, num_workers=4)
        input_dim_unsw = len(joblib.load(UNSW_DIR / 'columns_unsw.pkl'))
        
        results_unsw['Isolation Forest'] = evaluate_isoforest(UNSWDataset)
        
        model_mlp_unsw = MLPAutoencoder(input_dim_unsw, [64, 32], 16).to(DEVICE)
        if (MODEL_SAVE_DIR / 'best_mlp_ae_unsw.pth').exists():
            model_mlp_unsw.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_mlp_ae_unsw.pth', map_location=DEVICE))
            results_unsw['MLP-AE (15 epochs)'] = evaluate_autoencoder(model_mlp_unsw, unsw_test_loader)
        
        model_qkan_unsw = QKANAutoencoder(input_dim_unsw, [64, 32], 16).to(DEVICE)
        if (MODEL_SAVE_DIR / 'best_qkan_ae_unsw.pth').exists():
            model_qkan_unsw.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_qkan_ae_unsw.pth', map_location=DEVICE))
            results_unsw['QKAN-AE (15 epochs)'] = evaluate_autoencoder(model_qkan_unsw, unsw_test_loader)
        
        all_final_results['UNSW-NB15'] = results_unsw
    else:
        print("\nSkipping UNSW-NB15 experiments: Processed data not found.")

    # --- TẠO BẢNG ---
    create_latex_tables(all_final_results)

    # --- TẠO HÌNH ẢNH ---
    create_figures(model_qkan_ids)

    print("\n\nAll outputs generated successfully!")
