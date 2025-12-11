import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import random

# =====================================================
# 1. SET PATH DATASET DAN UKURAN CITRA
# =====================================================
# Jalur ini berfungsi ASALKAN folder 'fruits-360' sudah diekstrak di folder proyek Anda.
train_dir = "fruits-360/Training"
test_dir = "fruits-360/Test"

# Ukuran citra diatur 16x16 piksel
IMG_SIZE = 16 

# =====================================================
# 2. LOAD IMAGE FUNCTION (MODIFIED: Batas 15 Gambar per Kelas)
# =====================================================
# Batas diubah menjadi 15 gambar per kelas untuk meningkatkan akurasi.
def load_images_from_folder(folder_path, max_images_per_class=15): 
    X, y = [], []
    classes = sorted(os.listdir(folder_path))

    for label in classes:
        class_folder = os.path.join(folder_path, label)
        if not os.path.isdir(class_folder):
            continue
        
        print(f"Loading class: {label}")
        
        # Ambil daftar file
        image_list = os.listdir(class_folder)

        # Batasi iterasi hanya pada 15 gambar per kelas
        for img_name in tqdm(image_list[:max_images_per_class]): 
            img_path = os.path.join(class_folder, img_name)
            
            # Cek apakah itu file
            if not os.path.isfile(img_path):
                continue
                
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Preprocessing: Resize ke 16x16 piksel
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

# =====================================================
# 3. LOAD DATASET
# =====================================================
print("\nLoading TRAIN DATASET...")
# Total training samples awal: 131 kelas * 15 = 1965
X_train, y_train = load_images_from_folder(train_dir) 

print("\nLoading TEST DATASET...")
# Total test samples awal: 131 kelas * 15 = 1965
X_test, y_test = load_images_from_folder(test_dir)

# Normalisasi nilai piksel (0-255 menjadi 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# =====================================================
# 4. AUGMENTASI DATA
# =====================================================
def augment_images(X, y):
    X_aug, y_aug = [], []

    for img, label in zip(X, y):
        # Tambahkan gambar asli (sudah dinormalisasi)
        X_aug.append(img)
        y_aug.append(label)

        # Flip horizontal
        X_aug.append(cv2.flip(img, 1))
        y_aug.append(label)

        # Rotate 15 degrees
        center = (img.shape[1] // 2, img.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, 15, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        X_aug.append(rotated)
        y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)

print("\nApplying augmentation...")
# Data training total setelah augmentasi: 1965 samples * 3 = 5895 samples
X_aug, y_aug = augment_images(X_train, y_train) 

# Combine original + augmented
X_train = X_aug 
y_train = y_aug 

print(f"Total images after augmentation: {X_train.shape}")

# =====================================================
# 5. LABEL ENCODING + FLATTEN
# =====================================================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Flatten: Mengubah gambar 16x16x3 menjadi vektor 768 elemen
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

print("Final input shape:", X_train_flat.shape)

# =====================================================
# 6. TRAIN XGBOOST
# =====================================================
print("\nTraining XGBoost Model...")
# Inisialisasi model XGBoost untuk klasifikasi multi-kelas
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    objective="multi:softprob",
    num_class=len(le.classes_),
    tree_method="hist",
    random_state=42
)

model.fit(X_train_flat, y_train_enc)
print("Training complete.")

# =====================================================
# 7. EVALUATION
# =====================================================
# Memprediksi probabilitas dan mengambil kelas dengan probabilitas tertinggi
y_pred = np.argmax(model.predict_proba(X_test_flat), axis=1)

acc = accuracy_score(y_test_enc, y_pred)
print("\n=======================================")
print("          MODEL PERFORMANCE")
print("=======================================")
print(f"Accuracy : {acc*100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_, zero_division=0))

# =====================================================
# 8. CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_test_enc, y_pred)

plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=False, cmap="Blues") 
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show() 

# =====================================================
# 9. PLOT DISTRIBUSI KELAS
# =====================================================
plt.figure(figsize=(15, 5))
(unique, counts) = np.unique(y_train, return_counts=True)
plt.bar(unique, counts)
plt.xticks(rotation=90)
plt.title("Distribution of Classes (Train Data)")
plt.show() 

# =====================================================
# 10. VISUALISASI HASIL KLASIFIKASI (GAMBAR)
# =====================================================
print("\nShowing sample predictions...")

plt.figure(figsize=(15, 10))

# Ambil 10 sampel acak dari data test
indexes = random.sample(range(len(X_test)), 10)

for i, idx in enumerate(indexes):
    img = X_test[idx]
    true_label = le.classes_[y_test_enc[idx]]
    pred_label = le.classes_[y_pred[idx]]

    color = "green" if true_label == pred_label else "red"

    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)

plt.tight_layout()
plt.show()