import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ========== 1. Load dữ liệu ========== #
def load_images_labels(image_dir, label_path, image_size=(32, 32)):
    labels = np.load(label_path)
    images = []
    filenames = sorted(os.listdir(image_dir), key=lambda x: int(x.split(".")[0]))
    for fname in filenames:
        img = Image.open(os.path.join(image_dir, fname)).convert("L").resize(image_size)
        images.append(np.array(img) / 255.0)
    images = np.array(images)
    return images, labels[:len(images)]

# Đường dẫn
image_dir = "/content/drive/MyDrive/ColabNotebooks/nienluancoso1/output/dataset_10k"
label_path = "/content/drive/MyDrive/ColabNotebooks/nienluancoso1/output/y_train.npy"

# Load dữ liệu
print("🔍 Đang load dữ liệu...")
X, y_raw = load_images_labels(image_dir, label_path)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded)
print("✅ Dữ liệu đã load xong!")

# ========== Chuẩn bị dữ liệu ========== #
X_cnn = X.reshape(-1, 32, 32, 1)
X_rnn = X.reshape(-1, 32, 32)
X_flat = X.reshape(len(X), -1)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_categorical, test_size=0.2, random_state=42)
X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_rnn, y_categorical, test_size=0.2, random_state=42)
X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(X_flat, y_categorical, test_size=0.2, random_state=42)

# ========== 2. CNN ========== #
cnn = models.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_categorical.shape[1], activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Huấn luyện CNN...")
start_cnn = time.time()
cnn.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
cnn_time = time.time() - start_cnn
cnn_loss, cnn_acc = cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)

# ========== 3. RNN (LSTM) ========== #
rnn = models.Sequential([
    layers.Input(shape=(32, 32)),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_categorical.shape[1], activation='softmax')
])
rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Huấn luyện RNN...")
start_rnn = time.time()
rnn.fit(X_train_rnn, y_train_rnn, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
rnn_time = time.time() - start_rnn
rnn_loss, rnn_acc = rnn.evaluate(X_test_rnn, y_test_rnn, verbose=0)

# ========== 4. MLP ========== #
mlp = models.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(y_categorical.shape[1], activation='softmax')
])
mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Huấn luyện MLP...")
start_mlp = time.time()
mlp.fit(X_train_flat, y_train_flat, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
mlp_time = time.time() - start_mlp
mlp_loss, mlp_acc = mlp.evaluate(X_test_flat, y_test_flat, verbose=0)

# ========== 5. DNN ========== #
dnn = models.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(y_categorical.shape[1], activation='softmax')
])
dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Huấn luyện DNN...")
start_dnn = time.time()
dnn.fit(X_train_flat, y_train_flat, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
dnn_time = time.time() - start_dnn
dnn_loss, dnn_acc = dnn.evaluate(X_test_flat, y_test_flat, verbose=0)

# ========== 6. So sánh kết quả ========== #
print("\n🎯 === KẾT QUẢ TỔNG KẾT ===")
print(f"✅ CNN  | Accuracy: {cnn_acc*100:.2f}% | Time: {cnn_time:.2f}s")
print(f"🔁 RNN  | Accuracy: {rnn_acc*100:.2f}% | Time: {rnn_time:.2f}s")
print(f"🧠 MLP  | Accuracy: {mlp_acc*100:.2f}% | Time: {mlp_time:.2f}s")
print(f"📚 DNN  | Accuracy: {dnn_acc*100:.2f}% | Time: {dnn_time:.2f}s")

# Optional: Vẽ biểu đồ
plt.bar(['CNN', 'RNN', 'MLP', 'DNN'], [cnn_acc*100, rnn_acc*100, mlp_acc*100, dnn_acc*100], color=['skyblue', 'orange', 'purple', 'green'])
plt.ylabel("Accuracy (%)")
plt.title("So sánh mô hình học sâu")
plt.ylim(0, 100)
plt.grid(axis='y')
plt.show()
