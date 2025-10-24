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
import time
import sys
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

# Import các thành phần dùng chung từ utils.py
from utils import QKANAutoencoder, MLPAutoencoder, IntrusionDataset, UNSWDataset

# --- CẤU HÌNH ---
IDS2018_DIR = Path("./processed_data/")
UNSW_DIR = Path("./processed_data_unsw/")
MODEL_SAVE_DIR = Path("./models/")
FIGURES_DIR = Path("./figures/")
FIGURES_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DPI = 600
BATCH_SIZE = 4096 # Batch size lớn để đánh giá nhanh hơn

# --- CÀI ĐẶT STYLE & FONT CHO PLOT ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 15,
    'figure.titlesize': 20, 'font.family': 'serif', 'font.serif': ['Times New Roman']
})
warnings.filterwarnings("ignore", category=UserWarning)

# --- CÁC HÀM TIỆN ÍCH ĐÁNH GIÁ ---
def get_metrics(y_true, y_pred_binary, scores):
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    return {
        'F1-Score': f1_score(y_true, y_pred_binary, zero_division=0), 'AUC Score': roc_auc_score(y_true, scores),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0), 'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
    }

def evaluate_autoencoder(model, dataloader, model_name="Model"):
    model.to(DEVICE).eval()
    criterion = nn.MSELoss(reduction='none')
    all_errors, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {model_name}", leave=False):
            inputs = inputs.to(DEVICE)
            reconstructions = model(inputs)
            errors = criterion(reconstructions, inputs).mean(dim=1)
            all_errors.append(errors.cpu().numpy()); all_labels.append(labels.cpu().numpy())
    all_errors = np.concatenate(all_errors); all_labels = np.concatenate(all_labels)
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    y_pred_binary = (all_errors > best_threshold).astype(int)
    return get_metrics(all_labels, y_pred_binary, all_errors), (all_errors, all_labels, y_pred_binary)

def evaluate_isoforest(dataset_class, data_dir, dataset_name):
    print(f"Evaluating Isolation Forest on {dataset_name}...")
    test_df = dataset_class(data_dir, is_train=False).df
    train_df = dataset_class(data_dir, is_train=True).df
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
        for i, (inputs, _) in enumerate(dataloader):
            if i >= 50: break
            inputs = inputs.to(DEVICE)
            starter.record(); _ = model(inputs); ender.record()
            torch.cuda.synchronize(); timings.append(starter.elapsed_time(ender))
    avg_inference_time = np.mean(timings) if timings else 0
    return params, avg_inference_time

# --- CÁC HÀM TẠO TABLES ---
def create_latex_tables(all_results):
    print("\n" + "="*80 + "\n                      LATEX TABLES\n" + "="*80)
    for dataset_name in ["CSE-CIC-IDS2018", "UNSW-NB15"]:
        if dataset_name in all_results and all_results[dataset_name]:
            df = pd.DataFrame.from_dict(all_results[dataset_name], orient='index')
            df = df[['F1-Score', 'AUC Score', 'Precision', 'Recall']]
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    best_val = df[col].max()
                    df[col] = df[col].apply(lambda x: f"\\textbf{{{x:.3f}}}" if x == best_val else f"{x:.3f}")
            latex_string = df.to_latex(escape=False, column_format="@{}lcccc@{}}")
            print(f"\n% --- TABLE: Performance on {dataset_name} ---")
            print(f"\\begin{{table}}[ht]"); print(f"\\centering")
            print(f"\\caption{{Performance on the {dataset_name.replace('_', ' ')} Dataset. Metrics are for the 'Attack' class. Best results are in \\textbf{{bold}}.}}")
            print(f"\\label{{tab:{dataset_name.lower().replace('-', '')}}}")
            print(latex_string); print(f"\\end{{table}}")
    if "Computational Cost" in all_results:
        df_cost = pd.DataFrame.from_dict(all_results["Computational Cost"], orient='index')
        df_cost['Trainable Params'] = df_cost['Trainable Params'].apply(lambda x: f"{x/1e3:.1f}K")
        df_cost['Inference Time (ms/batch)'] = df_cost['Inference Time (ms/batch)'].apply(lambda x: f"{x:.2f}")
        latex_cost = df_cost.to_latex(escape=False, column_format="@{}lrr@{}}")
        print(f"\n% --- TABLE: Computational Cost ---")
        print(f"\\begin{{table}}[ht]"); print(f"\\centering")
        print(f"\\caption{{Computational cost comparison (Batch Size={BATCH_SIZE}).}}")
        print(f"\\label{{tab:cost}}"); print(latex_cost); print(f"\\end{{table}}")

# --- CÁC HÀM TẠO FIGURES ---
def create_all_figures(model_dict, results_dict):
    print("\n" + "="*80 + "\n                      GENERATING FIGURES\n" + "="*80)
    qkan_ids, mlp_ids = model_dict['qkan_ids2018'], model_dict['mlp_ids2018']
    qkan_unsw, _ = model_dict.get('qkan_unsw'), model_dict.get('mlp_unsw')
    
    # FIGURE 2: Convergence Plot
    try:
        mlp_hist = joblib.load(MODEL_SAVE_DIR / "mlp_ids2018_history.joblib")
        qkan_hist = joblib.load(MODEL_SAVE_DIR / "qkan_ids2018_history.joblib")
        epochs = np.arange(1, len(mlp_hist) + 1)
        plt.figure(figsize=(8, 6)); plt.plot(epochs, mlp_hist, 'o-', label='MLP-AE'); plt.plot(epochs, qkan_hist, 's-', label='QKAN-AE')
        plt.xlabel("Epoch"); plt.ylabel("Validation Loss (MSE)"); plt.title("Validation Loss Convergence on CSE-CIC-IDS2018")
        plt.yscale('log'); plt.xticks(epochs[::2 if len(epochs) > 10 else 1]); plt.legend(); plt.tight_layout()
        plt.savefig(FIGURES_DIR / "figure2_convergence.png", dpi=DPI, bbox_inches='tight'); plt.close()
        print("Generated Figure 2: Convergence Plot.")
    except Exception as e: print(f"Skipping Figure 2 (Convergence): {e}")

    # FIGURE 3: Reconstruction Error Distributions
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        for i, (name, data) in enumerate([("CSE-CIC-IDS2018", results_dict['data_ids2018']), ("UNSW-NB15", results_dict['data_unsw'])]):
            errors, labels, _ = data
            sns.kdeplot(errors[labels==0], label='Benign', fill=True, ax=axes[i], clip=(0, None))
            sns.kdeplot(errors[labels==1], label='Attack', fill=True, ax=axes[i], clip=(0, None))
            axes[i].set_xlabel("Reconstruction Error (MSE)"); axes[i].set_title(name)
            axes[i].set_xlim(0, max(np.quantile(errors[labels==0], 0.999), np.quantile(errors[labels==1], 0.98)))
        axes[0].set_ylabel("Density"); axes[0].legend()
        fig.suptitle("Figure 3: Reconstruction Error Distributions", fontsize=20, fontweight='bold')
        plt.savefig(FIGURES_DIR / "figure3_error_distributions.png", dpi=DPI, bbox_inches='tight'); plt.close(fig)
        print("Generated Figure 3: Error Distributions.")
    except Exception as e: print(f"Skipping Figure 3 (Error Distributions): {e}")
    
    # FIGURE 4: Interpretability Case Study
    try:
        columns = joblib.load(IDS2018_DIR / 'columns.pkl')
        feature_name = 'Flow IAT Max'; feature_index = columns.index(feature_name)
        input_dim = len(columns); encoder = qkan_ids.encoder; encoder.eval()
        loader = DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=BATCH_SIZE)
        sample_inputs, _ = next(iter(loader))
        x_grid = torch.linspace(sample_inputs[:, feature_index].min().item(), sample_inputs[:, feature_index].max().item(), 200).to(DEVICE)
        base_input = torch.zeros(1, input_dim).to(DEVICE); y_grid = []
        with torch.no_grad():
            for val in x_grid:
                inp = base_input.clone(); inp[0, feature_index] = val; y_grid.append(encoder.layers[0](inp)[0, 0].item())
        plt.figure(figsize=(8, 6)); plt.plot(x_grid.cpu().numpy(), y_grid, linewidth=3)
        plt.title(f"Learned Activation for '{feature_name}'"); plt.xlabel(f"Input Value (Scaled)"); plt.ylabel("Activation Output")
        plt.grid(True); plt.tight_layout()
        plt.savefig(FIGURES_DIR / "figure4_interpretability.png", dpi=DPI, bbox_inches='tight'); plt.close()
        print("Generated Figure 4: Interpretability Case Study.")
    except Exception as e: print(f"Skipping Figure 4 (Interpretability): {e}")

    # FIGURE 6: Confusion Matrices
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        _, labels_mlp, pred_mlp = results_dict['data_ids2018_mlp']
        _, labels_qkan, pred_qkan = results_dict['data_ids2018']
        cm_mlp = confusion_matrix(labels_qkan, pred_mlp)
        cm_qkan = confusion_matrix(labels_qkan, pred_qkan)
        sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', ax=axes[0], annot_kws={"size": 14})
        axes[0].set_title("MLP-AE Confusion Matrix"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
        sns.heatmap(cm_qkan, annot=True, fmt='d', cmap='Blues', ax=axes[1], annot_kws={"size": 14})
        axes[1].set_title("QKAN-AE Confusion Matrix"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
        axes[0].set_xticklabels(['Benign', 'Attack']); axes[0].set_yticklabels(['Benign', 'Attack'])
        axes[1].set_xticklabels(['Benign', 'Attack']); axes[1].set_yticklabels(['Benign', 'Attack'])
        fig.suptitle("Figure 6: Confusion Matrices on CSE-CIC-IDS2018", fontsize=20, fontweight='bold')
        plt.savefig(FIGURES_DIR / "figure6_confusion_matrices.png", dpi=DPI, bbox_inches='tight'); plt.close(fig)
        print("Generated Figure 6: Confusion Matrices.")
    except Exception as e: print(f"Skipping Figure 6 (Confusion Matrices): {e}")

# --- HÀM MAIN ĐIỀU PHỐI ---
if __name__ == '__main__':
    all_results = {}
    models_to_load = {}
    raw_results_for_figs = {}
    print("\n" + "="*80 + "\n            GENERATING ALL TABLES AND FIGURES\n" + "="*80)

    # --- THÍ NGHIỆM TRÊN CSE-CIC-IDS2018 ---
    if IDS2018_DIR.exists():
        print("\n--- Evaluating on CSE-CIC-IDS2018 ---")
        results_ids2018 = {}
        input_dim = len(joblib.load(IDS2018_DIR / 'columns.pkl'))
        loader = DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=BATCH_SIZE, num_workers=4)
        
        results_ids2018['Isolation Forest'] = evaluate_isoforest(IntrusionDataset, IDS2018_DIR, "CSE-CIC-IDS2018")
        
        model_mlp = MLPAutoencoder(input_dim, DEVICE, use_clamp=False).to(DEVICE)
        model_mlp.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_mlp_ids2018.pth', map_location=DEVICE))
        results_ids2018['MLP-AE'], raw_results_for_figs['data_ids2018_mlp'] = evaluate_autoencoder(model_mlp, loader, "MLP-AE on IDS2018")
        models_to_load['mlp_ids2018'] = model_mlp

        model_qkan = QKANAutoencoder(input_dim, DEVICE, use_clamp=True).to(DEVICE)
        model_qkan.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_qkan_ids2018.pth', map_location=DEVICE))
        results_ids2018['QKAN-AE'], raw_results_for_figs['data_ids2018'] = evaluate_autoencoder(model_qkan, loader, "QKAN-AE on IDS2018")
        models_to_load['qkan_ids2018'] = model_qkan
        
        all_results['CSE-CIC-IDS2018'] = results_ids2018
    
    # --- THÍ NGHIỆM TRÊN UNSW-NB15 ---
    if UNSW_DIR.exists():
        print("\n--- Evaluating on UNSW-NB15 ---")
        results_unsw = {}; input_dim = len(joblib.load(UNSW_DIR / 'columns_unsw.pkl'))
        loader = DataLoader(UNSWDataset(UNSW_DIR, is_train=False), batch_size=BATCH_SIZE, num_workers=4)
        results_unsw['Isolation Forest'] = evaluate_isoforest(UNSWDataset, UNSW_DIR, "UNSW-NB15")
        
        model_mlp_unsw = MLPAutoencoder(input_dim, DEVICE, use_clamp=False).to(DEVICE)
        model_mlp_unsw.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_mlp_unsw.pth', map_location=DEVICE))
        results_unsw['MLP-AE'], _ = evaluate_autoencoder(model_mlp_unsw, loader, "MLP-AE on UNSW")
        models_to_load['mlp_unsw'] = model_mlp_unsw

        model_qkan_unsw = QKANAutoencoder(input_dim, DEVICE, use_clamp=False).to(DEVICE)
        model_qkan_unsw.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_qkan_unsw.pth', map_location=DEVICE))
        results_unsw['QKAN-AE'], raw_results_for_figs['data_unsw'] = evaluate_autoencoder(model_qkan_unsw, loader, "QKAN-AE on UNSW")
        models_to_load['qkan_unsw'] = model_qkan_unsw
        all_results['UNSW-NB15'] = results_unsw
    
    # --- THU THẬP DỮ LIỆU CHO BẢNG COST ---
    cost_data = {}
    params_mlp, time_mlp = get_model_params_and_time(models_to_load['mlp_ids2018'], DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=BATCH_SIZE))
    params_qkan, time_qkan = get_model_params_and_time(models_to_load['qkan_ids2018'], DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=BATCH_SIZE))
    cost_data['MLP-AE'] = {'Trainable Params': params_mlp, 'Inference Time (ms/batch)': time_mlp}
    cost_data['QKAN-AE'] = {'Trainable Params': params_qkan, 'Inference Time (ms/batch)': time_qkan}
    all_results['Computational Cost'] = cost_data

    # --- TẠO BẢNG & HÌNH ẢNH ---
    create_latex_tables(all_results)
    create_all_figures(models_to_load, raw_results_for_figs)

    print("\n\nAll outputs generated successfully!")
