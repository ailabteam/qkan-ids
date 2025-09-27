import pandas as pd
import numpy as np
from pathlib import Path

# --- CẤU HÌNH ---
# Hãy đảm bảo file này tồn tại từ lần chạy preprocess thành công trước đó
FILE_TO_ANALYZE = Path("./processed_data/02-14-2018.csv")

def analyze_data(file_path):
    """Phân tích một file dữ liệu đã được xử lý."""
    
    if not file_path.exists():
        print(f"Lỗi: File '{file_path}' không tồn tại. Vui lòng chạy lại bước tiền xử lý trước đó.")
        return

    print(f"--- Bắt đầu phân tích file: {file_path.name} ---")
    
    try:
        df = pd.read_csv(file_path)
        features_df = df.drop(columns=['Label'])
        
        print(f"Đọc thành công {len(df)} dòng với {len(features_df.columns)} đặc trưng.")
        
        # 1. Sử dụng describe() để xem thống kê tổng quan
        print("\n--- 1. Thống kê mô tả (describe()) ---")
        # Sử dụng to_string() để đảm bảo tất cả các cột được hiển thị
        description = features_df.describe().to_string()
        print(description)
        
        # 2. Phân tích các giá trị CỰC ĐOAN
        print("\n--- 2. Phân tích các giá trị cực đoan ---")
        
        max_vals = features_df.max()
        min_vals = features_df.min()
        
        print("\n  Cột có giá trị LỚN NHẤT:")
        # Sắp xếp để xem 5 cột có giá trị max lớn nhất
        print(max_vals.sort_values(ascending=False).head(5))
        
        print("\n  Cột có giá trị NHỎ NHẤT (âm nhất):")
        # Sắp xếp để xem 5 cột có giá trị min nhỏ nhất
        print(min_vals.sort_values(ascending=True).head(5))
        
        # 3. Kiểm tra sự hiện diện của NaN hoặc Inf (dù không nên có)
        print("\n--- 3. Kiểm tra lại sự tồn tại của NaN / Inf ---")
        nan_check = features_df.isnull().sum().sum()
        inf_check = np.isinf(features_df).sum().sum()
        
        print(f"  Tổng số giá trị NaN: {nan_check}")
        print(f"  Tổng số giá trị Inf: {inf_check}")
        
        print("\n--- Phân tích hoàn tất ---")
        
        # Đưa ra nhận xét dựa trên kết quả
        print("\n>>> NHẬN XÉT SƠ BỘ:")
        if max_vals.max() > 100 or min_vals.min() < -100:
            print("  (!) Dải giá trị của dữ liệu rất lớn (vượt quá +/- 100) ngay cả sau khi dùng RobustScaler.")
            print("      Điều này có khả năng cao là nguyên nhân gây ra exploding loss.")
            print("      Gợi ý: Sử dụng một phương pháp chuẩn hóa phi tuyến mạnh hơn như QuantileTransformer.")
        else:
            print("  (+) Dải giá trị của dữ liệu có vẻ hợp lý. Vấn đề có thể không nằm ở việc chuẩn hóa.")
            
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")

if __name__ == "__main__":
    analyze_data(FILE_TO_ANALYZE)
