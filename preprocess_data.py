import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import joblib
import warnings

# Bỏ qua các cảnh báo không quan trọng
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# --- Cấu hình ---
DATA_DIR = Path("./cicids2018_data/")
PROCESSED_DIR = Path("./processed_data/")
PROCESSED_DIR.mkdir(exist_ok=True)

def preprocess_and_save(df, file_path, scaler, columns_to_keep, is_fitting_scaler=False):
    """Hàm phụ để xử lý và lưu DataFrame."""
    
    # 1. Tách đặc trưng và nhãn
    X = df.drop(columns=['Label'])
    y = df['Label']

    # 2. Giữ lại các cột đã được chọn từ file đầu tiên
    X = X[columns_to_keep]

    # 3. Scale dữ liệu
    if is_fitting_scaler:
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, PROCESSED_DIR / 'scaler.pkl')
        joblib.dump(columns_to_keep, PROCESSED_DIR / 'columns.pkl')
    else:
        X_scaled = scaler.transform(X)
        
    X_scaled_df = pd.DataFrame(X_scaled, columns=columns_to_keep, index=df.index)
    
    # 4. Chuyển đổi nhãn
    y_processed = y.apply(lambda x: 'Benign' if 'Benign' in x else 'Attack')
    
    # 5. Kết hợp lại và lưu
    processed_df = pd.concat([X_scaled_df, y_processed.reset_index(drop=True)], axis=1)
    output_path = PROCESSED_DIR / file_path.name
    processed_df.to_csv(output_path, index=False)
    print(f"  Saved {len(processed_df)} processed rows to {output_path}")


def main():
    """Hàm chính để chạy toàn bộ quá trình tiền xử lý."""
    
    # Xóa dữ liệu cũ
    print("Clearing previous processed data...")
    for f in PROCESSED_DIR.glob('*'):
        f.unlink()
        
    csv_files = sorted([f for f in DATA_DIR.glob("*.csv")])
    
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    # --- Xử lý file đầu tiên để thiết lập scaler và cột ---
    print("\n--- Step 1: Processing the first file to initialize scaler ---")
    first_file_path = csv_files[0]
    print(f"Reading {first_file_path.name}...")
    df_first = pd.read_csv(first_file_path)
    
    # Dọn dẹp tên cột
    df_first.columns = df_first.columns.str.strip()
    
    # Xử lý các giá trị đặc biệt
    df_first = df_first.drop(columns=['Timestamp'], errors='ignore')
    df_first.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Chuyển đổi sang số, ép lỗi thành NaN
    feature_cols = [col for col in df_first.columns if col != 'Label']
    for col in feature_cols:
        df_first[col] = pd.to_numeric(df_first[col], errors='coerce')
    
    # Loại bỏ các dòng/cột có vấn đề
    initial_rows = len(df_first)
    df_first.dropna(inplace=True)
    print(f"  Dropped {initial_rows - len(df_first)} rows with NaN values.")
    
    # Loại bỏ các cột có phương sai bằng 0 (cột hằng số)
    X_temp = df_first.drop(columns=['Label'])
    variance = X_temp.var()
    columns_to_keep = variance[variance > 0].index.tolist()
    print(f"  Keeping {len(columns_to_keep)} features with non-zero variance.")
    
    scaler = RobustScaler()
    preprocess_and_save(df_first, first_file_path, scaler, columns_to_keep, is_fitting_scaler=True)

    # --- Xử lý các file còn lại ---
    print("\n--- Step 2: Processing remaining files ---")
    for file_path in tqdm(csv_files[1:], desc="Processing files"):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        df = df.drop(columns=['Timestamp'], errors='ignore')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        for col in feature_cols: # Dùng feature_cols từ file đầu
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)
        
        if not df.empty:
            preprocess_and_save(df, file_path, scaler, columns_to_keep)
        else:
            print(f"Skipping empty file: {file_path.name}")

    print("\n--- Preprocessing complete! ---")
    print(f"Processed data saved in: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
