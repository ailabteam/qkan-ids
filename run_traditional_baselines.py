# --- CẬP NHẬT run_traditional_baselines.py ---
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from pathlib import Path
import joblib

def run_baselines():
    df_train = pd.read_parquet("./processed_data_unsw/train.parquet")
    df_test = pd.read_parquet("./processed_data_unsw/test.parquet")
    
    X_train = df_train.drop(columns=['label']).values
    X_test = df_test.drop(columns=['label']).values
    y_test = df_test['label'].values

    summary_results = {}

    # 1. Isolation Forest
    print("Running Isolation Forest...")
    model = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train)
    scores = -model.decision_function(X_test)
    summary_results['IF'] = {'auc': roc_auc_score(y_test, scores)}

    # 2. SGD One-Class SVM (Nhanh cho dữ liệu lớn)
    print("Running OC-SVM...")
    model = SGDOneClassSVM(random_state=42)
    model.fit(X_train[df_train['label']==0]) # Chỉ fit trên benign
    scores = -model.decision_function(X_test)
    summary_results['OC-SVM'] = {'auc': roc_auc_score(y_test, scores)}

    # 3. LOF (Reviewer 3 yêu cầu - Dùng sampling vì 2 triệu dòng sẽ tràn RAM)
    print("Running LOF (on 50k samples for estimation)...")
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    # Lấy mẫu 50k để chạy trong thời gian thực tế
    X_train_sample = X_train[np.random.choice(len(X_train), 50000, replace=False)]
    model.fit(X_train_sample)
    scores = -model.decision_function(X_test)
    summary_results['LOF'] = {'auc': roc_auc_score(y_test, scores)}

    print("\n--- TRADITIONAL BASELINE RESULTS ---")
    for name, res in summary_results.items():
        print(f"{name}: AUC = {res['auc']:.4f}")
    
    joblib.dump(summary_results, "./models/traditional_baselines_final.joblib")

if __name__ == "__main__":
    run_baselines()
