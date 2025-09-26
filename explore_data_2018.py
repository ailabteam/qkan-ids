import pandas as pd
import os
from pathlib import Path

# Cập nhật đường dẫn đến thư mục dữ liệu mới
DATA_DIR = Path("./cicids2018_data/")

def explore_dataset(data_dir):
    """Hàm để khám phá bộ dữ liệu CSE-CIC-IDS2018."""
    
    if not data_dir.exists():
        print(f"Lỗi: Thư mục '{data_dir}' không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
        return

    # 1. Liệt kê tất cả các file CSV
    csv_files = sorted([f for f in data_dir.glob("*.csv")])
    print("--- Tìm thấy các file CSV sau: ---")
    if not csv_files:
        print("Không tìm thấy file CSV nào!")
        return
        
    for f in csv_files:
        print(f.name)
    print("-" * 35)

    # 2. Đọc thử file đầu tiên để phân tích
    # Chọn một file có kích thước vừa phải để đọc nhanh
    # Ví dụ: '02-14-2018.csv'
    file_to_analyze = data_dir / '02-14-2018.csv'
    if not file_to_analyze.exists():
        print(f"File {file_to_analyze.name} không tồn tại, chọn file đầu tiên trong danh sách.")
        file_to_analyze = csv_files[0]
        
    print(f"\n--- Phân tích file: {file_to_analyze.name} ---\n")
    
    try:
        # Đọc một lượng nhỏ dòng để xem trước, vì file có thể rất lớn
        df = pd.read_csv(file_to_analyze)

        # Tên cột trong bộ này cũng có thể có khoảng trắng
        df.columns = df.columns.str.strip()
        
        # 3. In thông tin cơ bản
        print("--- Thông tin DataFrame (df.info()) ---")
        df.info()
        print("\n--- 5 dòng đầu tiên (df.head()) ---")
        print(df.head())
        
        # 4. Kiểm tra cột Label
        if 'Label' in df.columns:
            print("\n--- Phân phối của các nhãn (Label) ---")
            print(df['Label'].value_counts())
        else:
            print("\nCảnh báo: Không tìm thấy cột 'Label' trong file.")

    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file: {e}")


if __name__ == "__main__":
    explore_dataset(DATA_DIR)
