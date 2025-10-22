import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm
import joblib
import warnings

# --- Cấu hình ---
DATA_DIR = Path("./unsw_nb15_data/")
PROCESSED_DIR = Path("./processed_data_unsw/")
PROCESSED_DIR.mkdir(exist_ok=True)

def main():
    print("--- Preprocessing UNSW-NB15 Dataset ---")
    
    for f in PROCESSED_DIR.glob('*'):
        f.unlink()
        
    all_files = sorted(list(DATA_DIR.glob('UNSW-NB15_*.csv')))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        df = pd.concat((pd.read_csv(f, header=None) for f in all_files), ignore_index=True)

    try:
        features_df = pd.read_csv(DATA_DIR / 'NUSW-NB15_features.csv', encoding='latin-1')
        column_names = features_df['Name'].tolist()
        df.columns = column_names
    except Exception as e:
        print(f"Lỗi khi đọc file features: {e}. Sẽ thoát.")
        return

    df.columns = df.columns.str.strip().str.lower() 

    cols_to_drop = ['id', 'attack_cat']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"Dropped columns: {existing_cols_to_drop}")

    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        if col == 'service' and '-' in df[col].unique():
            df[col] = df[col].replace('-', 'other')
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)

    # === DÒNG SỬA LỖI ĐƯỢC THÊM VÀO ĐÂY ===
    # Loại bỏ bất kỳ dòng nào có giá trị 'label' bị thiếu hoặc không hợp lệ TRƯỚC KHI chuyển đổi
    df.dropna(subset=['label'], inplace=True)
    df['label'] = pd.to_numeric(df['label'], errors='coerce') # Chuyển sang số, lỗi thành NaN
    df.dropna(subset=['label'], inplace=True) # Xóa các dòng lỗi một lần nữa

    # Bây giờ, việc chuyển sang int sẽ an toàn
    df['label'] = df['label'].astype(int)

    y = df['label']
    X = df.drop('label', axis=1)
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    print("Fitting and transforming data with QuantileTransformer...")
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X)-1), random_state=42)
    X_scaled = scaler.fit_transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    processed_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    
    output_path = PROCESSED_DIR / "unsw_nb15_processed.csv"
    processed_df.to_csv(output_path, index=False)

    joblib.dump(scaler, PROCESSED_DIR / 'scaler_unsw.pkl')
    joblib.dump(X.columns.tolist(), PROCESSED_DIR / 'columns_unsw.pkl')
    
    print(f"\nPreprocessing complete! Saved processed data to {output_path}")
    print(f"Number of features after one-hot encoding: {len(X.columns)}")

if __name__ == "__main__":
    main()
