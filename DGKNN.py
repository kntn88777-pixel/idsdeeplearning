import os
import time
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
        images.append(np.array(img) / 255.0)  # normalize

    images = np.array(images).reshape(len(images), -1)  # Flatten ảnh thành vector
    return images, labels[:len(images)]

# ===========================
# 2. Chuẩn bị dữ liệu
# ===========================
image_dir = "/content/drive/MyDrive/ColabNotebooks/nienluancoso1/output/dataset_10k"
label_path = "/content/drive/MyDrive/ColabNotebooks/nienluancoso1/output/y_train.npy"

X, y_raw = load_images_labels(image_dir, label_path)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ===========================
# 3. Huấn luyện KNN
# ===========================
knn = KNeighborsClassifier(n_neighbors=3)

start_train = time.time()
knn.fit(X_train, y_train)
end_train = time.time()

train_time = end_train - start_train
print(f"⏱️ Thời gian huấn luyện KNN: {train_time:.2f} giây")

# ===========================
# 4. Đánh giá mô hình
# ===========================
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Độ chính xác trên tập test: {accuracy * 100:.2f}%")

# ===========================
# 5. Dự đoán mẫu và đo thời gian phản hồi
# ===========================
sample = X_test[0:1]

start_pred = time.time()
predicted_class = knn.predict(sample)
end_pred = time.time()

predicted_label = label_encoder.inverse_transform(predicted_class)
print(f"📌 Dự đoán lớp: {predicted_label[0]}")
print(f"⏱️ Thời gian đưa ra phán đoán: {(end_pred - start_pred) * 1000:.2f} ms")

# ===========================
# 6. Vẽ confusion matrix
# ===========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.title("🔍 Confusion Matrix - KNN")
plt.show()
