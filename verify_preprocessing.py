import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
import warnings

warnings.filterwarnings('ignore')

# --- CẤU HÌNH THÍ NGHIỆM ---
# Sử dụng file dữ liệu thô
RAW_DATA_FILE = Path("./cicids2018_data/02-14-2018.csv") 
# Cột có vấn đề nhất mà chúng ta đã tìm thấy
COLUMN_TO_VERIFY = "Idle Max"
# Số lượng dòng để đọc làm mẫu
N_SAMPLES = 100000

def verify_transformer():
    """
    Thực hiện thí nghiệm kiểm chứng thu nhỏ để xem hiệu quả của QuantileTransformer.
    """
    if not RAW_DATA_FILE.exists():
        print(f"Lỗi: File dữ liệu thô '{RAW_DATA_FILE}' không tồn tại.")
        return

    print(f"--- Đang đọc {N_SAMPLES} dòng mẫu từ file: {RAW_DATA_FILE.name} ---")
    df = pd.read_csv(RAW_DATA_FILE, nrows=N_SAMPLES)
    
    # 1. Dọn dẹp dữ liệu mẫu
    df.columns = df.columns.str.strip()
    if COLUMN_TO_VERIFY not in df.columns:
        print(f"Lỗi: Không tìm thấy cột '{COLUMN_TO_VERIFY}' trong file.")
        return
        
    # Chuyển cột cần kiểm tra sang dạng số, loại bỏ các giá trị không hợp lệ
    df[COLUMN_TO_VERIFY] = pd.to_numeric(df[COLUMN_TO_VERIFY], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[COLUMN_TO_VERIFY], inplace=True)
    
    # Lấy dữ liệu của cột đó. Cần reshape để đưa vào scaler.
    data_column = df[[COLUMN_TO_VERIFY]].values
    
    print(f"\n--- 1. PHÂN TÍCH 'TRƯỚC' KHI BIẾN ĐỔI ---")
    print(f"Thống kê của cột '{COLUMN_TO_VERIFY}' (dữ liệu gốc đã làm sạch):")
    print(pd.DataFrame(data_column).describe().to_string())
    
    # 2. Áp dụng QuantileTransformer
    print("\n--- 2. ÁP DỤNG QuantileTransformer ---")
    # Sử dụng các tham số giống như trong script tiền xử lý đầy đủ
    transformer = QuantileTransformer(
        output_distribution='normal', 
        n_quantiles=1000, # Dùng 1000 quantiles cho độ chính xác
        random_state=42
    )
    
    print("  Đang fit và transform dữ liệu...")
    transformed_data = transformer.fit_transform(data_column)
    
    # 3. Phân tích dữ liệu SAU khi biến đổi
    print(f"\n--- 3. PHÂN TÍCH 'SAU' KHI BIẾN ĐỔI ---")
    print(f"Thống kê của cột '{COLUMN_TO_VERIFY}' (dữ liệu đã qua QuantileTransformer):")
    print(pd.DataFrame(transformed_data).describe().to_string())

    print("\n" + "="*50)
    print(">>> KẾT LUẬN KIỂM CHỨNG <<<")
    
    before_stats = pd.DataFrame(data_column).describe()
    after_stats = pd.DataFrame(transformed_data).describe()

    print(f"  - Dải giá trị TRƯỚC: [{before_stats.loc['min'][0]:.2f}, {before_stats.loc['max'][0]:.2e}]")
    print(f"  - Dải giá trị SAU:   [{after_stats.loc['min'][0]:.2f}, {after_stats.loc['max'][0]:.2f}]")
    
    if after_stats.loc['max'][0] < 10 and after_stats.loc['min'][0] > -10:
        print("\n  (+) THÀNH CÔNG! QuantileTransformer đã 'thuần hóa' thành công dữ liệu.")
        print("      Dữ liệu đầu ra có phân phối chuẩn, dải giá trị nhỏ và ổn định.")
        print("      Chúng ta có thể tự tin tiến hành chạy lại toàn bộ quá trình tiền xử lý.")
    else:
        print("\n  (-) THẤT BẠI! Vẫn còn vấn đề với phương pháp chuẩn hóa.")
        
    print("="*50)

if __name__ == "__main__":
    verify_transformer()
