import torch
import sys

print("--- Bắt đầu kiểm tra môi trường ---")
print(f"Phiên bản Python: {sys.version}")

try:
    print(f"Phiên bản PyTorch: {torch.__version__}")
    is_cuda = torch.cuda.is_available()
    print(f"PyTorch có thể sử dụng CUDA (GPU): {is_cuda}")

    if not is_cuda:
        print("\n!!! LỖI: PyTorch không tìm thấy CUDA. Vui lòng kiểm tra lại bước cài đặt PyTorch.")
    else:
        gpu_count = torch.cuda.device_count()
        print(f"Số lượng GPU tìm thấy: {gpu_count}")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

        # Kiểm tra các thư viện khác
        print("\n--- Kiểm tra các thư viện phụ ---")
        import pandas as pd
        print(f"Pandas: OK (version {pd.__version__})")
        import sklearn
        print(f"Scikit-learn: OK (version {sklearn.__version__})")
        import qkan
        print(f"qkan: OK")
        import kan
        print(f"pykan: OK")
        import tqdm
        print(f"tqdm: OK")
        
        print("\n>>> CHÚC MỪNG: Môi trường đã được cài đặt thành công và sẵn sàng để chạy lại các thí nghiệm!")

except ImportError as e:
    print(f"\n!!! LỖI: Không thể import thư viện. Lỗi: {e}")
    print("Vui lòng kiểm tra lại bước `pip install -r requirements.txt`.")
except Exception as e:
    print(f"\n!!! LỖI không xác định: {e}")

print("\n--- Kết thúc kiểm tra ---")
