import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# --- Cấu hình ---
DATA_DIR = Path("./cicids2018_data/")
PROCESSED_DIR = Path("./processed_data/")
PROCESSED_DIR.mkdir(exist_ok=True)

def preprocess_file(file_path, scaler=None, is_fitting_scaler=False):
    """Tiền xử lý một file CSV duy nhất, xử lý các dòng lỗi."""
    
    print(f"Processing {file_path.name}...")
    
    # 1. Đọc file CSV với low_memory=False để xử lý DtypeWarning tốt hơn
    df = pd.read_csv(file_path, low_memory=False)
    
    # 2. Dọn dẹp tên cột
    df.columns = df.columns.str.strip()
    
    # 3. Loại bỏ cột không cần thiết
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])
    
    # 4. Chuyển đổi các cột sang dạng số, các giá trị lỗi sẽ thành NaN
    # Lấy danh sách các cột đặc trưng (trừ 'Label')
    feature_cols = [col for col in df.columns if col != 'Label']
    for col in tqdm(feature_cols, desc=f"  Converting columns in {file_path.name}", leave=False):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 5. Xử lý giá trị vô cực và NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Bây giờ, dropna sẽ loại bỏ cả các giá trị thiếu ban đầu và các dòng bị lỗi (chứa header)
    df.dropna(inplace=True)

    if df.empty:
        print(f"Warning: DataFrame is empty after cleaning {file_path.name}. Skipping file.")
        return scaler
        
    # 6. Tách đặc trưng (X) và nhãn (y)
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # 7. Chuẩn hóa dữ liệu số (Scaling)
    if is_fitting_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, PROCESSED_DIR / 'scaler.pkl')
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for subsequent files.")
        X_scaled = scaler.transform(X)
        
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # 8. Chuyển đổi nhãn
    y_processed = y.apply(lambda x: 'Benign' if x == 'Benign' else 'Attack')
    
    # Kết hợp lại X và y
    processed_df = pd.concat([X_scaled, y_processed.reset_index(drop=True)], axis=1)
    
    # 9. Lưu file đã xử lý
    output_path = PROCESSED_DIR / file_path.name
    processed_df.to_csv(output_path, index=False)
    print(f"Saved processed file to {output_path}")
    
    return scaler

def main():
    """Hàm chính để chạy toàn bộ quá trình tiền xử lý."""
    
    # Xóa các file đã xử lý trước đó để chạy lại từ đầu
    print("Clearing previous processed data...")
    for f in PROCESSED_DIR.glob('*'):
        f.unlink()
        
    csv_files = sorted([f for f in DATA_DIR.glob("*.csv")])
    
    if not csv_files:
        print(f"Không tìm thấy file CSV nào trong {DATA_DIR}")
        return
        
    print("--- Bước 1: Fitting scaler trên file đầu tiên ---")
    scaler = preprocess_file(csv_files[0], is_fitting_scaler=True)
    
    print("\n--- Bước 2: Transforming các file còn lại ---")
    for file_path in tqdm(csv_files[1:], desc="Processing all files"):
        preprocess_file(file_path, scaler=scaler)
        
    print("\n--- Hoàn tất quá trình tiền xử lý! ---")
    print(f"Dữ liệu đã xử lý được lưu tại thư mục: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
