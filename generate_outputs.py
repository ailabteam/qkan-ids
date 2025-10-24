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
# Đặt trong try-except để báo lỗi rõ ràng nếu file utils.py thiếu
try:
    from utils import QKANAutoencoder, MLPAutoencoder, IntrusionDataset, UNSWDataset
except ImportError as e:
    print(f"FATAL ERROR: Could not import from utils.py. Please ensure the file exists and is correct.")
    print(f"Details: {e}")
    sys.exit(1)

# --- CẤU HÌNH ---
IDS2018_DIR = Path("./processed_data/")
UNSW_DIR = Path("./processed_data_unsw/")
MODEL_SAVE_DIR = Path("./models/")
FIGURES_DIR = Path("./figures/")
FIGURES_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DPI = 600
BATCH_SIZE = 4096

# --- CÀI ĐẶT STYLE & FONT CHO PLOT ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    plt.rcParams.update({
        'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 15,
        'figure.titlesize': 20, 'font.family': 'serif', 'font.serif': ['Times New Roman']
    })
except Exception as e:
    print(f"Warning: Could not set plot style. Matplotlib/Seaborn might have issues. {e}")

warnings.filterwarnings("ignore", category=UserWarning)

# --- CÁC HÀM TIỆN ÍCH ĐÁNH GIÁ ---

def get_metrics(y_true, y_pred_binary, scores):
    """Tính toán các chỉ số đánh giá từ kết quả."""
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    # Chuyển y_true sang 0 (benign) và 1 (attack) để nhất quán
    y_true_binary = (y_true != 0).astype(int)
    return {
        'F1-Score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        'AUC Score': roc_auc_score(y_true_binary, scores),
        'Precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
    }

def evaluate_autoencoder(model, dataloader, model_name="Model"):
    """Chạy đánh giá cho một mô hình autoencoder và trả về kết quả."""
    model.to(DEVICE).eval()
    criterion = nn.MSELoss(reduction='none')
    all_errors, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {model_name}", leave=False):
            inputs = inputs.to(DEVICE)
            reconstructions = model(inputs)
            # Tính MSE trên từng sample trong batch
            errors = criterion(reconstructions, inputs).mean(dim=1)
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels)
    
    from sklearn.metrics import precision_recall_curve
    # Tìm ngưỡng tối ưu dựa trên F1-score cho lớp Attack (label=1)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Thêm kiểm tra để tránh lỗi index rỗng
    if len(thresholds) == 0:
        best_threshold = 0.5
    else:
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]
        
    y_pred_binary = (all_errors > best_threshold).astype(int)
    
    # Trả về cả metrics và dữ liệu thô để vẽ hình
    return get_metrics(all_labels, y_pred_binary, all_errors), (all_errors, all_labels, y_pred_binary)

def evaluate_isoforest(dataset_class, data_dir, dataset_name):
    """Chạy đánh giá cho Isolation Forest."""
    print(f"Evaluating Isolation Forest on {dataset_name}...")
    test_df = dataset_class(data_dir, is_train=False).df
    train_df = dataset_class(data_dir, is_train=True).df
    
    label_col = 'label' if 'label' in test_df.columns else 'Label'
    # Chuyển label sang 0 (benign) và 1 (attack)
    y_true = test_df[label_col].apply(lambda x: 0 if (x == 'Benign' or x == 0) else 1)

    X_test = test_df.drop(columns=[label_col])
    X_train = train_df.drop(columns=[label_col])
    
    clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    # Huấn luyện trên mẫu dữ liệu benign
    clf.fit(X_train.sample(n=min(len(X_train), 250000), random_state=42))
    
    scores = -clf.decision_function(X_test) # Điểm càng cao càng bất thường
    y_pred_binary = (clf.predict(X_test) == -1).astype(int) # -1 là anomaly
    
    return get_metrics(y_true, y_pred_binary, scores)

def get_model_params_and_time(model, dataloader):
    """Đo số lượng tham số và tốc độ suy luận."""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Khởi động GPU
    for i, (inputs, _) in enumerate(dataloader):
        if i >= 2: break
        _ = model(inputs.to(DEVICE))

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= 50: # Đo trên 50 batch
                break
            inputs = inputs.to(DEVICE)
            starter.record()
            _ = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender)) # ms
    
    avg_inference_time = np.mean(timings) if timings else 0
    return params, avg_inference_time


# --- CÁC HÀM TẠO TABLES ---

def create_latex_tables(all_results):
    """
    Tự động tạo và in ra các bảng LaTeX từ dictionary kết quả.
    """
    print("\n" + "="*80)
    print("                      LATEX TABLES")
    print("="*80)

    # --- Bảng 1 & 2: Main Performance ---
    for dataset_name in ["CSE-CIC-IDS2018", "UNSW-NB15"]:
        if dataset_name in all_results and all_results[dataset_name]:
            df = pd.DataFrame.from_dict(all_results[dataset_name], orient='index')
            
            # Sắp xếp lại thứ tự cột và hàng cho logic
            df = df[['F1-Score', 'AUC Score', 'Precision', 'Recall']]
            model_order = [m for m in ['Isolation Forest', 'MLP-AE', 'QKAN-AE'] if m in df.index]
            df = df.loc[model_order]

            # Tìm giá trị tốt nhất ở mỗi cột để in đậm
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    best_val = df[col].max()
                    # Định dạng lại các số và in đậm giá trị tốt nhất
                    df[col] = df[col].apply(lambda x: f"\\textbf{{{x:.3f}}}" if x == best_val else f"{x:.3f}")
            
            latex_string = df.to_latex(escape=False, column_format="@{}lcccc@{}}")
            
            print(f"\n% --- TABLE: Performance on {dataset_name} ---")
            print(f"\\begin{{table}}[ht]")
            print(f"\\centering")
            print(f"\\caption{{Performance on the {dataset_name.replace('_', ' ')} Dataset. Metrics are for the 'Attack' class. Best results are in \\textbf{{bold}}.}}")
            print(f"\\label{{tab:{dataset_name.lower().replace('-', '')}}}")
            print(latex_string)
            print(f"\\end{{table}}")

    # --- Bảng 4: Computational Cost ---
    if "Computational Cost" in all_results and all_results["Computational Cost"]:
        df_cost = pd.DataFrame.from_dict(all_results["Computational Cost"], orient='index')
        
        # Định dạng các cột
        df_cost['Trainable Params'] = df_cost['Trainable Params'].apply(lambda x: f"{x/1e3:.1f}K")
        df_cost['Inference Time (ms/batch)'] = df_cost['Inference Time (ms/batch)'].apply(lambda x: f"{x:.2f}")
        
        latex_cost = df_cost.to_latex(escape=False, column_format="@{}lrr@{}}")
        
        print(f"\n% --- TABLE: Computational Cost ---")
        print(f"\\begin{{table}}[ht]")
        print(f"\\centering")
        print(f"\\caption{{Computational cost comparison (Batch Size={BATCH_SIZE}).}}")
        print(f"\\label{{tab:cost}}")
        print(latex_cost)
        print(f"\\end{{table}}")

# --- CÁC HÀM TẠO FIGURES ---

def create_all_figures(model_dict, results_dict):
    """
    Tự động tạo và lưu tất cả các hình ảnh cần thiết cho bài báo.
    """
    print("\n" + "="*80)
    print("                      GENERATING FIGURES")
    print("="*80)

    # Lấy ra các model và dữ liệu cần thiết từ dictionary
    qkan_ids = model_dict.get('qkan_ids2018')
    qkan_unsw = model_dict.get('qkan_unsw')

    # --- FIGURE 2: Validation Loss Convergence ---
    try:
        mlp_hist = joblib.load(MODEL_SAVE_DIR / "mlp_ids2018_history.joblib")
        qkan_hist = joblib.load(MODEL_SAVE_DIR / "qkan_ids2018_history.joblib")
        epochs = np.arange(1, len(mlp_hist) + 1)
        
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, mlp_hist, 'o-', label='MLP-AE')
        plt.plot(epochs, qkan_hist, 's-', label='QKAN-AE')
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss (MSE)")
        plt.title("Validation Loss Convergence on CSE-CIC-IDS2018")
        plt.yscale('log')
        plt.xticks(epochs[::2 if len(epochs) > 10 else 1])
        plt.legend()
        plt.tight_layout()
        
        save_path = FIGURES_DIR / "figure2_convergence.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Generated Figure 2: {save_path}")
        plt.close()
    except Exception as e:
        print(f"Skipping Figure 2 (Convergence): {e}")

    # --- FIGURE 3: Reconstruction Error Distributions ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Panel (a): CSE-CIC-IDS2018
        errors_ids, labels_ids, _ = results_dict['data_ids2018']
        sns.kdeplot(errors_ids[labels_ids==0], label='Benign', fill=True, ax=axes[0], clip=(0, None), lw=2.5)
        sns.kdeplot(errors_ids[labels_ids==1], label='Attack', fill=True, ax=axes[0], clip=(0, None), lw=2.5)
        axes[0].set_xlabel("Reconstruction Error (MSE)")
        axes[0].set_title("CSE-CIC-IDS2018")
        axes[0].set_xlim(0, max(np.quantile(errors_ids[labels_ids==0], 0.999), np.quantile(errors_ids[labels_ids==1], 0.98)))
        
        # Panel (b): UNSW-NB15
        errors_unsw, labels_unsw, _ = results_dict['data_unsw']
        sns.kdeplot(errors_unsw[labels_unsw==0], label='Normal', fill=True, ax=axes[1], clip=(0, None), lw=2.5)
        sns.kdeplot(errors_unsw[labels_unsw==1], label='Attack', fill=True, ax=axes[1], clip=(0, None), lw=2.5)
        axes[1].set_xlabel("Reconstruction Error (MSE)")
        axes[1].set_title("UNSW-NB15")
        axes[1].set_xlim(0, max(np.quantile(errors_unsw[labels_unsw==0], 0.999), np.quantile(errors_unsw[labels_unsw==1], 0.98)))

        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[1].legend()
        fig.suptitle("Figure 3: Reconstruction Error Distributions", fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        
        save_path = FIGURES_DIR / "figure3_error_distributions.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Generated Figure 3: {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Skipping Figure 3 (Error Distributions): {e}")

    # --- FIGURE 4: Interpretability Case Study ---
    try:
        columns = joblib.load(IDS2018_DIR / 'columns.pkl')
        feature_name = 'Flow IAT Max'
        feature_index = columns.index(feature_name)
        input_dim = len(columns)
        encoder = qkan_ids.encoder
        encoder.eval()
        
        loader = DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=BATCH_SIZE)
        sample_inputs, _ = next(iter(loader))
        
        x_grid = torch.linspace(sample_inputs[:, feature_index].min().item(), sample_inputs[:, feature_index].max().item(), 200).to(DEVICE)
        base_input = torch.zeros(1, input_dim).to(DEVICE)
        y_grid = []
        
        with torch.no_grad():
            for val in x_grid:
                inp = base_input.clone()
                inp[0, feature_index] = val
                activation_output = encoder.layers[0](inp)
                y_grid.append(activation_output[0, 0].item()) # Ảnh hưởng đến node 0 của lớp tiếp theo
                
        plt.figure(figsize=(8, 6))
        plt.plot(x_grid.cpu().numpy(), y_grid, linewidth=3)
        plt.title(f"Learned Activation for '{feature_name}'")
        plt.xlabel(f"Input Value (Scaled)")
        plt.ylabel("Activation Output")
        plt.grid(True)
        plt.tight_layout()
        
        save_path = FIGURES_DIR / "figure4_interpretability.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Generated Figure 4: {save_path}")
        plt.close()
    except Exception as e:
        print(f"Skipping Figure 4 (Interpretability): {e}")

    # --- FIGURE 6: Confusion Matrices ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel (a): MLP-AE Confusion Matrix
        _, labels_mlp, pred_mlp = results_dict['data_ids2018_mlp']
        cm_mlp = confusion_matrix(labels_mlp, pred_mlp)
        sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', ax=axes[0], annot_kws={"size": 16}, cbar=False)
        axes[0].set_title("MLP-AE")
        
        # Panel (b): QKAN-AE Confusion Matrix
        _, labels_qkan, pred_qkan = results_dict['data_ids2018']
        cm_qkan = confusion_matrix(labels_qkan, pred_qkan)
        sns.heatmap(cm_qkan, annot=True, fmt='d', cmap='Blues', ax=axes[1], annot_kws={"size": 16}, cbar=False)
        axes[1].set_title("QKAN-AE")

        for ax in axes:
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_xticklabels(['Benign', 'Attack'])
            ax.set_yticklabels(['Benign', 'Attack'], va='center')

        fig.suptitle("Figure 6: Confusion Matrices on CSE-CIC-IDS2018", fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = FIGURES_DIR / "figure6_confusion_matrices.png"
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Generated Figure 6: {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Skipping Figure 6 (Confusion Matrices): {e}")


# --- HÀM MAIN ĐIỀU PHỐI ---

if __name__ == '__main__':
    # Dictionary để lưu tất cả kết quả
    all_results = {}
    # Dictionary để lưu các model đã tải để tái sử dụng
    loaded_models = {}
    # Dictionary để lưu dữ liệu thô (errors, labels) cho việc vẽ hình
    raw_data_for_figs = {}

    print("\n" + "="*80)
    print("            GENERATING ALL TABLES AND FIGURES FOR THE PAPER")
    print("="*80)

    # --- GIAI ĐOẠN 1: THU THẬP KẾT QUẢ TRÊN CSE-CIC-IDS2018 ---
    if IDS2018_DIR.exists() and len(list(IDS2018_DIR.glob('*.csv'))) > 0:
        print("\n--- Processing CSE-CIC-IDS2018 ---")
        results_ids2018 = {}
        input_dim = len(joblib.load(IDS2018_DIR / 'columns.pkl'))
        loader = DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=BATCH_SIZE, num_workers=4)

        # Isolation Forest
        results_ids2018['Isolation Forest'] = evaluate_isoforest(IntrusionDataset, IDS2018_DIR, "CSE-CIC-IDS2018")

        # MLP-AE
        model_mlp = MLPAutoencoder(input_dim, DEVICE).to(DEVICE)
        model_mlp.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_mlp_ids2018.pth', map_location=DEVICE))
        results_ids2018['MLP-AE'], raw_data_for_figs['data_ids2018_mlp'] = evaluate_autoencoder(model_mlp, loader, "MLP-AE on IDS2018")
        loaded_models['mlp_ids2018'] = model_mlp

        # QKAN-AE (model tốt nhất - có thể là 5 hoặc 15 epochs tùy bạn chọn)
        # Chúng ta sẽ dùng bản 15-epoch để nhất quán, nhưng dùng bản 5-epoch cho hình 3 & 4
        model_qkan_15e = QKANAutoencoder(input_dim, DEVICE, use_clamp=True).to(DEVICE)
        model_qkan_15e.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_qkan_ids2018.pth', map_location=DEVICE))
        results_ids2018['QKAN-AE'], raw_data_for_figs['data_ids2018'] = evaluate_autoencoder(model_qkan_15e, loader, "QKAN-AE on IDS2018")
        loaded_models['qkan_ids2018'] = model_qkan_15e
        
        all_results['CSE-CIC-IDS2018'] = results_ids2018
    else:
        print("\nSkipping CSE-CIC-IDS2018: Processed data not found.")

    # --- GIAI ĐOẠN 2: THU THẬP KẾT QUẢ TRÊN UNSW-NB15 ---
    if UNSW_DIR.exists() and (UNSW_DIR / "unsw_nb15_processed.csv").exists():
        print("\n--- Processing UNSW-NB15 ---")
        results_unsw = {}
        input_dim = len(joblib.load(UNSW_DIR / 'columns_unsw.pkl'))
        loader = DataLoader(UNSWDataset(UNSW_DIR, is_train=False), batch_size=BATCH_SIZE, num_workers=4)

        results_unsw['Isolation Forest'] = evaluate_isoforest(UNSWDataset, UNSW_DIR, "UNSW-NB15")

        model_mlp_unsw = MLPAutoencoder(input_dim, DEVICE).to(DEVICE)
        model_mlp_unsw.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_mlp_unsw.pth', map_location=DEVICE))
        results_unsw['MLP-AE'], _ = evaluate_autoencoder(model_mlp_unsw, loader, "MLP-AE on UNSW")
        loaded_models['mlp_unsw'] = model_mlp_unsw

        model_qkan_unsw = QKANAutoencoder(input_dim, DEVICE, use_clamp=False).to(DEVICE)
        model_qkan_unsw.load_state_dict(torch.load(MODEL_SAVE_DIR / 'best_qkan_unsw.pth', map_location=DEVICE))
        results_unsw['QKAN-AE'], raw_data_for_figs['data_unsw'] = evaluate_autoencoder(model_qkan_unsw, loader, "QKAN-AE on UNSW")
        loaded_models['qkan_unsw'] = model_qkan_unsw
        
        all_results['UNSW-NB15'] = results_unsw
    else:
        print("\nSkipping UNSW-NB15: Processed data not found.")
        
    # --- GIAI ĐOẠN 3: THU THẬP DỮ LIỆU CHO BẢNG COST ---
    if 'mlp_ids2018' in loaded_models and 'qkan_ids2018' in loaded_models:
        print("\n--- Collecting Computational Cost Data ---")
        cost_data = {}
        loader_ids = DataLoader(IntrusionDataset(IDS2018_DIR, is_train=False), batch_size=BATCH_SIZE)
        
        params_mlp, time_mlp = get_model_params_and_time(loaded_models['mlp_ids2018'], loader_ids)
        params_qkan, time_qkan = get_model_params_and_time(loaded_models['qkan_ids2018'], loader_ids)
        
        cost_data['MLP-AE'] = {'Trainable Params': params_mlp, 'Inference Time (ms/batch)': time_mlp}
        cost_data['QKAN-AE'] = {'Trainable Params': params_qkan, 'Inference Time (ms/batch)': time_qkan}
        all_results['Computational Cost'] = cost_data

    # --- GIAI ĐOẠN 4: TẠO "THÀNH PHẨM" ---
    if all_results:
        create_latex_tables(all_results)
        create_all_figures(loaded_models, raw_data_for_figs)
        print("\n\nAll outputs generated successfully!")
    else:
        print("\nNo results were collected. Exiting.")
        
