import joblib
from pathlib import Path

MODEL_SAVE_DIR = Path("./models/")
MODEL_SAVE_DIR.mkdir(exist_ok=True)

# --- DỮ LIỆU TỪ MLP-AE (15 Epochs) ---
mlp_val_loss = [
    0.058073,
    0.032417,
    0.022902,
    0.019226,
    0.016976,
    0.015338,
    0.013764,
    0.012801,
    0.011870,
    0.011214,
    0.010531,
    0.010044,
    0.009646,
    0.009273,
    0.009024
]

# --- DỮ LIỆU TỪ QKAN-AE v2 (15 Epochs) ---
qkan_val_loss = [
    0.030354,
    0.018894,
    0.015300,
    0.013509,
    0.012048,
    0.011419,
    0.010658,
    0.009882,
    0.009341,
    0.009004,
    0.008557,
    0.008219,
    0.008316, # Epoch 13 loss tăng nhẹ
    0.007803,
    0.007555
]

# Lưu các file history
if len(mlp_val_loss) > 0:
    mlp_history_path = MODEL_SAVE_DIR / "MLPAutoencoder_history.joblib"
    joblib.dump(mlp_val_loss, mlp_history_path)
    print(f"Đã tạo file history cho MLP tại: {mlp_history_path}")

if len(qkan_val_loss) > 0:
    qkan_history_path = MODEL_SAVE_DIR / "QKANAutoencoder_history.joblib"
    joblib.dump(qkan_val_loss, qkan_history_path)
    print(f"Đã tạo file history cho QKAN tại: {qkan_history_path}")
