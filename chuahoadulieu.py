# 🚀 TIỀN XỬ LÝ DỮ LIỆU IDS + CÂN BẰNG TRÊN GOOGLE COLAB

# 1. Cài thư viện nếu cần
!pip install -q imbalanced-learn

# 2. Import thư viện
import pandas as pd
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from datetime import datetime
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

# 3. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 4. Đường dẫn đến thư mục chứa các file CSV
csv_folder = '/content/drive/MyDrive/ColabNotebooks/nienluancoso1/DATA/CSE-IDS2018'
print(f"[1] Đọc dữ liệu từ thư mục: {csv_folder}")

# 5. Đọc & gộp các file CSV
df_list = []
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(csv_folder, filename)
        print(f"    → Đang đọc: {filename}")
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df.drop(columns=['Flow ID', 'Timestamp'], inplace=True, errors='ignore')
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            if 'Label' in df.columns:
                df_list.append(df)
        except Exception as e:
            print(f"    ❌ Lỗi đọc {filename}: {e}")

if not df_list:
    raise ValueError("❌ Không có file CSV hợp lệ!")

df = pd.concat(df_list, ignore_index=True)
print(f"[2] Tổng số dòng sau khi gộp: {df.shape[0]}, số cột: {df.shape[1]}")

gc.collect()

# 6. Tách đặc trưng và nhãn
if 'Label' not in df.columns:
    raise ValueError("❌ Không tìm thấy cột 'Label'!")

X = df.drop(columns=['Label'])
y = df['Label']

# 7. Mã hoá các cột dạng chuỗi
print("[3] Mã hoá các cột dạng chuỗi...")
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    print(f"    → {col} đã mã hoá.")

# 8. Mã hoá nhãn
print("[4] Mã hoá nhãn Label...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 9. Hiển thị phân phối trước cân bằng
print("[📊] Phân phối nhãn trước khi cân bằng:")
class_counts = pd.Series(y).value_counts()
class_labels = label_encoder.inverse_transform(class_counts.index)

plt.figure(figsize=(10, 5))
sns.barplot(x=class_labels, y=class_counts.values, palette='viridis')
plt.title("Số lượng mẫu theo lớp trước khi cân bằng", fontsize=14)
plt.xlabel("Lớp (Label)")
plt.ylabel("Số lượng mẫu")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 10. Cân bằng dữ liệu bằng RandomOverSampler
print("[⚖️] Cân bằng dữ liệu bằng RandomOverSampler...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# In phân phối sau khi cân bằng
print("[📊] Phân phối nhãn sau khi cân bằng:")
balanced_counts = Counter(y_resampled)
for idx, count in balanced_counts.items():
    label = label_encoder.inverse_transform([idx])[0]
    print(f"    {label}: {count} mẫu")

# 11. Chuẩn hoá đặc trưng
print("[5] Chuẩn hoá đặc trưng bằng StandardScaler...")
X_scaled = StandardScaler().fit_transform(X_resampled)

# 12. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print(f"[6] Train: {X_train.shape}, Test: {X_test.shape}")

# 13. Lưu ra CSV và NPY
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = '/content/drive/MyDrive/ColabNotebooks/nienluancoso1/output'
os.makedirs(output_dir, exist_ok=True)

# Ghi train
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['Label'] = y_train
train_csv = os.path.join(output_dir, f"train_data_{timestamp}.csv")
train_df.to_csv(train_csv, index=False)
print(f"✅ Đã ghi tập huấn luyện: {train_csv}")

# Ghi test
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['Label'] = y_test
test_csv = os.path.join(output_dir, f"test_data_{timestamp}.csv")
test_df.to_csv(test_csv, index=False)
print(f"✅ Đã ghi tập kiểm tra: {test_csv}")

# Ghi file numpy
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)
print("✅ Đã ghi file X_train.npy, X_test.npy, y_train.npy, y_test.npy")
