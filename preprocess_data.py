import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm
import joblib
import warnings

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# --- CẤU HÌNH ---
DATA_DIR = Path("./cicids2018_data/")
PROCESSED_DIR = Path("./processed_data/")
PROCESSED_DIR.mkdir(exist_ok=True)

def clean_dataframe(df, feature_cols):
    df.columns = df.columns.str.strip()
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def main():
    print("--- Preprocessing CSE-CIC-IDS2018 Dataset ---")
    
    print("Clearing previous processed data...")
    for f in PROCESSED_DIR.glob('*'):
        f.unlink()
        
    csv_files = sorted([f for f in DATA_DIR.glob("*.csv")])
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}. Please download the dataset first.")
        return

    print("\n--- Step 1: Finding the common set of columns ---")
    common_columns = None
    for file_path in tqdm(csv_files, desc="Reading headers"):
        df_header = pd.read_csv(file_path, nrows=0)
        current_columns = set([col.strip() for col in df_header.columns])
        if common_columns is None:
            common_columns = current_columns
        else:
            common_columns.intersection_update(current_columns)
    
    COLS_TO_DROP = {'Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port'}
    feature_columns = sorted(list(common_columns - COLS_TO_DROP - {'Label'}))
    print(f"Found {len(feature_columns)} common feature columns to be used.")

    print("\n--- Step 2: Sampling data to fit the scaler ---")
    sample_dfs = []
    for file_path in tqdm(csv_files, desc="Sampling data"):
        try:
            # usecols ensures we only read necessary columns, saving memory
            df_sample_chunk = pd.read_csv(file_path, usecols=feature_columns + ['Label'], nrows=50000)
            sample_dfs.append(df_sample_chunk)
        except Exception:
            continue
    df_sample = pd.concat(sample_dfs, ignore_index=True)

    df_sample = clean_dataframe(df_sample, feature_columns)
    
    variance = df_sample[feature_columns].var(numeric_only=True)
    columns_to_keep = variance[variance > 1e-6].index.tolist()
    print(f"Keeping {len(columns_to_keep)} features with significant variance.")
    joblib.dump(columns_to_keep, PROCESSED_DIR / 'columns.pkl')

    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(df_sample)//10, 1000), 10), random_state=42)
    print("\n--- Step 3: Fitting the scaler ---")
    scaler.fit(df_sample[columns_to_keep])
    joblib.dump(scaler, PROCESSED_DIR / 'scaler.pkl')

    print("\n--- Step 4: Transforming and saving all files ---")
    for file_path in tqdm(csv_files, desc="Processing files"):
        try:
            df_full = pd.read_csv(file_path, usecols=columns_to_keep + ['Label'])
        except ValueError: # Happens if a file is missing a column from columns_to_keep (unlikely but safe)
            df_full = pd.read_csv(file_path)

        df_full = clean_dataframe(df_full, columns_to_keep)
        
        if df_full.empty:
            continue

        X_full = df_full[columns_to_keep]
        y_full = df_full['Label'].apply(lambda x: 'Benign' if 'Benign' in x else 'Attack')
        
        X_scaled = scaler.transform(X_full)
        X_scaled_df = pd.DataFrame(X_scaled, columns=columns_to_keep)
        
        processed_df = pd.concat([X_scaled_df, y_full.reset_index(drop=True)], axis=1)
        output_path = PROCESSED_DIR / file_path.name
        processed_df.to_csv(output_path, index=False)
        
    print("\n--- Preprocessing complete for CSE-CIC-IDS2018! ---")

if __name__ == "__main__":
    main()
