import numpy as np
import os
from PIL import Image

# Đường dẫn đến file X_train.npy
X_train_path = "/content/drive/MyDrive/ColabNotebooks/nienluancoso1/output/X_train.npy"

# Tải dữ liệu X_train
X_train = np.load(X_train_path)

# Chọn 10.000 vector đầu tiên
X_sample = X_train[:10000]

# Đường dẫn lưu ảnh
output_dir = "/content/drive/MyDrive/ColabNotebooks/nienluancoso1/output/dataset_10k"
os.makedirs(output_dir, exist_ok=True)

# Kích thước ảnh mong muốn (ví dụ: 32x32)
image_size = (32, 32)

# Hàm chuyển vector thành ảnh và lưu
for i, vector in enumerate(X_sample):
    # Chuyển vector thành ma trận vuông (pad nếu cần)
    size = image_size[0] * image_size[1]
    if len(vector) < size:
        vector = np.pad(vector, (0, size - len(vector)))
    elif len(vector) > size:
        vector = vector[:size]
    
    image_array = vector.reshape(image_size)
    image = Image.fromarray((image_array * 255).astype(np.uint8))  # Convert sang uint8 cho ảnh

    # Lưu ảnh dạng PNG (hoặc .jpg nếu bạn thích)
    image.save(os.path.join(output_dir, f"{i}.png"))

print("✅ Đã tạo xong 10.000 ảnh đầu tiên!")
