# 1. KÜTÜPHANELER
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 2. VERİ HAZIRLIĞI
IMG_SIZE = 128
data = []
labels = []
dataset_path = 'teeth_dataset/Training'

for label in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, label)
    if not os.path.isdir(class_path):  # 🔥 sadece klasörse işleme devam et
        continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(0 if label.lower() == 'without_caries' else 1)

X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(labels)

# 3. VERİYİ BÖL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL OLUŞTUR
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. EĞİT
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

# 6. KAYDET
model.save("model/caries_model.h5")
print("Model başarıyla kaydedildi! ✅")