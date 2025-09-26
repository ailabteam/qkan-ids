import torch
from qkan import QKAN

# --- Kiểm tra PyTorch và GPU ---
print(f"PyTorch version: {torch.__version__}")
is_cuda_available = torch.cuda.is_available()
print(f"CUDA available: {is_cuda_available}")

if is_cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    device = "cuda"
else:
    print("CUDA not available, using CPU.")
    device = "cpu"

print("-" * 30)

# --- Kiểm tra QKAN có thể khởi tạo trên GPU không ---
try:
    print("Attempting to initialize a small QKAN model on the selected device...")
    # Tạo một mô hình QKAN nhỏ ví dụ: 2 input -> 5 hidden -> 1 output
    model = QKAN([2, 5, 1], device=device)
    print("QKAN model initialized successfully!")
    
    # Tạo một tensor dummy và đẩy lên GPU
    dummy_input = torch.randn(10, 2).to(device)
    output = model(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")
    print("Forward pass successful!")
    print("\nEnvironment setup is complete and correct! We are ready to proceed.")

except Exception as e:
    print("\nAn error occurred during the test:")
    print(e)
    print("Please check the installation steps.")
