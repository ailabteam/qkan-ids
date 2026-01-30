import pandas as pd
from pathlib import Path

def check_data():
    data_dir = Path("./unsw_nb15_data/")
    files = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
    features_file = "NUSW-NB15_features.csv"
    
    total_rows = 0
    print("--- CHƯƠNG TRÌNH KIỂM TRA DỮ LIỆU UNSW-NB15 ---")
    
    # 1. Kiểm tra file features
    if (data_dir / features_file).exists():
        feat_df = pd.read_csv(data_dir / features_file, encoding='latin-1')
        print(f"[OK] Tìm thấy file features. Số lượng đặc trưng gốc: {len(feat_df)}")
    else:
        print(f"[ERROR] Thiếu file {features_file}!")

    # 2. Kiểm tra các file CSV chính
    for f_name in files:
        f_path = data_dir / f_name
        if f_path.exists():
            # Đếm số dòng mà không load cả file vào RAM (để tiết kiệm bộ nhớ)
            count = sum(1 for line in open(f_path, 'r', encoding='latin-1'))
            print(f"[OK] {f_name}: {count} dòng")
            total_rows += count
            
            # Kiểm tra thử 1 dòng để xem delimiter
            sample = pd.read_csv(f_path, nrows=1, header=None)
            print(f"     -> Cấu trúc cột mẫu: {sample.shape[1]} cột")
        else:
            print(f"[ERROR] Không tìm thấy file {f_name}")

    print("-" * 40)
    print(f"TỔNG CỘNG SỐ DÒNG: {total_rows}")
    print(f"MỤC TIÊU TRONG PAPER: ~2,540,044 dòng")
    
    if abs(total_rows - 2540044) < 10000: # Cho phép sai số nhỏ do header
        print("\n=> KẾT LUẬN: DATASET ĐÚNG VỚI MÔ TẢ TRONG PAPER. SẴN SÀNG TIẾN HÀNH!")
    else:
        print("\n=> KẾT LUẬN: CÓ SỰ SAI LỆCH SỐ DÒNG. CẦN KIỂM TRA LẠI FILE TẢI VỀ.")

if __name__ == "__main__":
    check_data()
