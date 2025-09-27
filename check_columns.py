import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path("./cicids2018_data/")

def check_all_columns():
    """
    Đọc header của tất cả các file CSV và in ra danh sách cột để so sánh.
    """
    print("--- Bắt đầu kiểm tra sự nhất quán của các cột trong tất cả các file CSV ---")
    
    csv_files = sorted([f for f in DATA_DIR.glob("*.csv")])
    if not csv_files:
        print(f"Không tìm thấy file CSV nào trong thư mục: {DATA_DIR}")
        return

    all_columns = {}
    for file_path in csv_files:
        try:
            # Chỉ đọc dòng đầu tiên (header) để lấy tên cột, rất nhanh
            df_header = pd.read_csv(file_path, nrows=0) 
            # Dọn dẹp khoảng trắng trong tên cột
            columns = tuple(sorted([col.strip() for col in df_header.columns]))
            all_columns[file_path.name] = columns
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path.name}: {e}")

    # So sánh các bộ cột
    unique_column_sets = set(all_columns.values())
    
    print(f"\n>>> Tìm thấy {len(unique_column_sets)} bộ cột khác nhau trong tổng số {len(csv_files)} file.")

    if len(unique_column_sets) == 1:
        print("\n(+) TỐT: Tất cả các file CSV đều có cùng một bộ cột.")
        print(f"    Số lượng cột: {len(list(unique_column_sets)[0])}")
    else:
        print("\n(!) CẢNH BÁO: Các file CSV KHÔNG có cùng một bộ cột!")
        for i, col_set in enumerate(unique_column_sets):
            print(f"\n--- Bộ cột #{i+1} (có {len(col_set)} cột) ---")
            files_with_this_set = [name for name, cols in all_columns.items() if cols == col_set]
            print(f"  Các file sử dụng bộ cột này: {files_with_this_set}")
            # In ra một vài cột để xem ví dụ
            # print(f"  Ví dụ cột: {list(col_set)[:5]}")
            
    # Tìm sự khác biệt
    if len(unique_column_sets) > 1:
        print("\n--- Phân tích sự khác biệt giữa các bộ cột ---")
        sets = list(unique_column_sets)
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                set1 = set(sets[i])
                set2 = set(sets[j])
                diff1 = set1 - set2
                diff2 = set2 - set1
                if diff1 or diff2:
                    print(f"\nSự khác biệt giữa Bộ #{i+1} và Bộ #{j+1}:")
                    if diff1:
                        print(f"  Chỉ có trong Bộ #{i+1}: {sorted(list(diff1))}")
                    if diff2:
                        print(f"  Chỉ có trong Bộ #{j+1}: {sorted(list(diff2))}")


if __name__ == "__main__":
    check_all_columns()
