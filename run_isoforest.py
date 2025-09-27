import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from tqdm import tqdm
import joblib

# --- Cấu hình ---
PROCESSED_DIR = Path("./processed_data/")

def run_isolation_forest():
    """
    Huấn luyện và đánh giá mô hình Isolation Forest.
    """
    print("--- Running Isolation Forest Baseline ---")

    # 1. Tải dữ liệu
    print("Loading all processed data...")
    # Tải toàn bộ dữ liệu vì Isolation Forest không cần tách train/test theo cách của AE
    all_files = sorted([f for f in PROCESSED_DIR.glob("*.csv")])
    df_list = [pd.read_csv(file) for file in tqdm(all_files, desc="Loading files")]
    df = pd.concat(df_list, ignore_index=True)
    df.dropna(inplace=True)

    # Lấy số cột feature đã lưu
    try:
        columns_to_keep = joblib.load(PROCESSED_DIR / 'columns.pkl')
        X = df[columns_to_keep]
    except (FileNotFoundError, KeyError):
        print("Warning: columns.pkl not found or columns mismatch. Inferring from data.")
        X = df.drop(columns=['Label'])
        
    y_true = df['Label'].apply(lambda x: 1 if x == 'Benign' else -1) # y_true: 1 (normal), -1 (anomaly)
    y_true_metrics = df['Label'].apply(lambda x: 1 if x == 'Benign' else 0) # y_true cho metrics: 1 (Benign), 0 (Attack)

    print(f"Data loaded: {len(df)} samples.")

    # 2. Huấn luyện mô hình
    print("\nTraining Isolation Forest model...")
    # contamination='auto' là một thiết lập tốt. n_jobs=-1 để dùng tất cả các nhân CPU.
    # Lấy một mẫu nhỏ để huấn luyện cho nhanh, vì IF không cần toàn bộ dữ liệu
    n_samples_train = min(len(X), 250000)
    X_train_sample = X.sample(n=n_samples_train, random_state=42)

    clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    clf.fit(X_train_sample)

    print("Model training complete.")

    # 3. Đánh giá mô hình trên toàn bộ dữ liệu
    print("\nEvaluating model on the full dataset...")
    # predict() trả về 1 cho inliers (normal) và -1 cho outliers (anomaly)
    y_pred = clf.predict(X)
    
    # decision_function() trả về điểm bất thường. Cần đảo dấu để tính AUC
    # (điểm càng thấp càng bất thường, trong khi AUC mong đợi điểm càng cao càng bất thường)
    anomaly_scores = -clf.decision_function(X)

    # Chuyển đổi y_pred cho phù hợp với metrics
    # y_pred_metrics: 1 (Benign), 0 (Attack)
    y_pred_metrics = np.array([1 if x == 1 else 0 for x in y_pred])

    # 4. In kết quả
    print("\n--- Evaluation Results ---")
    
    # Sử dụng target_names để báo cáo dễ đọc hơn
    report = classification_report(y_true_metrics, y_pred_metrics, target_names=['Attack', 'Benign'])
    print(report)
    
    # Tính F1-Score riêng cho lớp "Attack" (lớp thiểu số)
    f1_attack = f1_score(y_true_metrics, y_pred_metrics, pos_label=0)
    print(f"F1-Score (Attack Class): {f1_attack:.4f}")

    # AUC Score
    auc = roc_auc_score(y_true_metrics, anomaly_scores)
    print(f"AUC Score: {auc:.4f}")

if __name__ == "__main__":
    run_isolation_forest()
