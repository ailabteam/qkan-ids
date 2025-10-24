import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm
import joblib
import warnings

# --- CẤU HÌNH ---
DATA_DIR = Path("./unsw_nb15_data/")
PROCESSED_DIR = Path("./processed_data_unsw/")
PROCESSED_DIR.mkdir(exist_ok=True)

def main():
    print("--- Preprocessing UNSW-NB15 Dataset ---")
    
    # Xóa dữ liệu cũ
    for f in PROCESSED_DIR.glob('*'):
        f.unlink()
        
    all_files = sorted(list(DATA_DIR.glob('UNSW-NB15_*.csv')))
    if not all_files:
        print(f"No UNSW-NB15_*.csv files found in {DATA_DIR}. Please download the dataset first.")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        # Dữ liệu gốc không có header, nên đọc với header=None
        df = pd.concat((pd.read_csv(f, header=None) for f in all_files), ignore_index=True)

    # Gán tên cột từ file features.csv
    try:
        features_df = pd.read_csv(DATA_DIR / 'NUSW-NB15_features.csv', encoding='latin-1')
        column_names = features_df['Name'].tolist()
        if len(df.columns) == len(column_names):
            df.columns = column_names
        else:
              print("Warning: Column count mismatch. Using default column names.")
    except Exception as e:
        print(f"Warning: Could not read features file: {e}. Using default column names.")
        
    # Dọn dẹp tên cột và loại bỏ các cột không cần thiết
    df.columns = df.columns.str.strip().str.lower()
    cols_to_drop = [col for col in ['id', 'attack_cat'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")

    # Xử lý các cột Categorical
    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('-', 'other') # Xử lý giá trị '-'
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)

    # Xử lý cột 'label'
    if 'label' not in df.columns:
        print("Error: 'label' column not found. Cannot proceed.")
        return
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # Tách features và labels
    y = df['label']
    X = df.drop('label', axis=1)
    
    # Đảm bảo tất cả các cột feature đều là số
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True) # Điền 0 vào các giá trị còn thiếu

    # Chuẩn hóa bằng QuantileTransformer
    print("Fitting and transforming data with QuantileTransformer...")
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X)-1), random_state=42)
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Kết hợp lại và lưu
    processed_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    output_path = PROCESSED_DIR / "unsw_nb15_processed.csv"
    processed_df.to_csv(output_path, index=False)

    # Lưu scaler và danh sách cột
    joblib.dump(scaler, PROCESSED_DIR / 'scaler_unsw.pkl')
    joblib.dump(X.columns.tolist(), PROCESSED_DIR / 'columns_unsw.pkl')
    
    print(f"\nPreprocessing complete for UNSW-NB15! Saved to {output_path}")
    print(f"Number of features after one-hot encoding: {len(X.columns)}")

if __name__ == "__main__":
    main()

