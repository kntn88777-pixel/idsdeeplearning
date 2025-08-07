import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ===========================
# 1. Đọc ảnh và nhãn
# ===========================
def load_images_labels(image_dir, label_path, image_size=(32, 32)):
    labels = np.load(label_path)

    images = []
    filenames = sorted(os.listdir(image_dir), key=lambda x: int(x.split(".")[0]))
    for fname in filenames:
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).convert("L").resize(image_size)
        images.append(np.array(img) / 255.0)

    images = np.array(images)
    return images, labels[:len(images)]

# ===========================
# 2. Chuẩn bị dữ liệu
# ===========================
image_dir = "/content/drive/MyDrive/ColabNotebooks/nienluancoso1/output/dataset_10k"
label_path = "/content/drive/MyDrive/ColabNotebooks/nienluancoso1/output/y_train.npy"

X, y_raw = load_images_labels(image_dir, label_path)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded)

# RNN cần input dạng (samples, time_steps, features)
# Ta xem mỗi dòng ảnh là 1 bước thời gian, mỗi cột là đặc trưng (32x32) → (32 time_steps, 32 features)
X = X.reshape(-1, 32, 32)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# ===========================
# 3. Xây mô hình RNN (LSTM)
# ===========================
model = models.Sequential([
    layers.LSTM(64, input_shape=(32, 32)),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===========================
# 4. Huấn luyện & đo thời gian
# ===========================
start_train = time.time()
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
end_train = time.time()

train_time = end_train - start_train
print(f"⏱️ Thời gian huấn luyện: {train_time:.2f} giây")

# ===========================
# 5. Đánh giá mô hình
# ===========================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Độ chính xác trên tập test: {accuracy * 100:.2f}%")

# ===========================
# 6. Dự đoán mẫu & đo thời gian
# ===========================
sample = X_test[0:1]
start_pred = time.time()
prediction = model.predict(sample)
end_pred = time.time()

predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
print(f"📌 Dự đoán lớp: {predicted_class[0]}")
print(f"⏱️ Thời gian đưa ra phán đoán: {(end_pred - start_pred) * 1000:.2f} ms")

# ===========================
# 7. Ma trận nhầm lẫn
# ===========================
y_test_labels = np.argmax(y_test, axis=1)
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.title("🔁 Confusion Matrix - RNN (LSTM)")
plt.show()
