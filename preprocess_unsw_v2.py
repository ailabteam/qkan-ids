import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import joblib
import warnings

# --- CẤU HÌNH ---
DATA_DIR = Path("./unsw_nb15_data/")
PROCESSED_DIR = Path("./processed_data_unsw/")
PROCESSED_DIR.mkdir(exist_ok=True)

def main():
    print("--- 1. LOADING DATA ---")
    all_files = [DATA_DIR / f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
    features_df = pd.read_csv(DATA_DIR / 'NUSW-NB15_features.csv', encoding='latin-1')
    column_names = [col.strip().lower() for col in features_df['Name'].tolist()]

    df_list = []
    for f in all_files:
        print(f"Reading {f.name}...")
        temp_df = pd.read_csv(f, header=None, encoding='latin-1', low_memory=False)
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    df.columns = column_names
    print(f"Total raw samples: {len(df)}")

    print("\n--- 2. CLEANING & NUMERIC CONVERSION ---")
    # Loại bỏ các cột không liên quan
    cols_to_drop = ['srcip', 'dstip', 'sport', 'dsport', 'attack_cat']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Xử lý các cột categorical riêng (proto, service, state)
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().replace(['-', ''], 'other')

    # ÉP KIỂU SỐ (QUAN TRỌNG): Chuyển các ô có khoảng trắng ' ' thành NaN
    non_cat_cols = [c for c in df.columns if c not in cat_cols and c != 'label']
    for col in non_cat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Lấp đầy NaN và vô hạn bằng 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    print("\n--- 3. SPLITTING (BEFORE TRANSFORMATION) ---")
    df_train_raw, df_test_raw = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Train size: {len(df_train_raw)}, Test size: {len(df_test_raw)}")

    print("\n--- 4. CATEGORICAL ENCODING ---")
    # One-hot encoding đồng nhất cho cả 2 tập
    combined = pd.concat([df_train_raw, df_test_raw])
    combined = pd.get_dummies(combined, columns=cat_cols)
    
    df_train = combined.iloc[:len(df_train_raw)].copy()
    df_test = combined.iloc[len(df_train_raw):].copy()

    print("\n--- 5. CHỐNG LEAKAGE: FITTING SCALER ---")
    # Fit scaler chỉ trên Benign của Train set
    train_benign = df_train[df_train['label'] == 0].drop(columns=['label'])
    
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
    scaler.fit(train_benign)
    print("Scaler fitted successfully on clean numeric data.")

    print("\n--- 6. TRANSFORMING & SAVING (PARQUET) ---")
    feature_cols = [c for c in df_train.columns if c != 'label']
    
    def transform_and_save(target_df, filename):
        X = target_df[feature_cols].values
        X_scaled = scaler.transform(X)
        
        final_df = pd.DataFrame(X_scaled, columns=feature_cols)
        final_df['label'] = target_df['label'].values
        
        output_path = PROCESSED_DIR / f"{filename}.parquet"
        final_df.to_parquet(output_path, compression='snappy')
        print(f"Saved: {output_path} ({len(final_df)} samples)")

    transform_and_save(df_train, "train")
    transform_and_save(df_test, "test")

    # Lưu metadata
    joblib.dump(scaler, PROCESSED_DIR / 'scaler_unsw.pkl')
    joblib.dump(feature_cols, PROCESSED_DIR / 'columns_unsw.pkl')
    print("\nPreprocessing Complete!")

if __name__ == "__main__":
    main()
