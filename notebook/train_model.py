import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 2. VERİ HAZIRLIĞI
IMG_SIZE = 224  # Görüntü boyutunu 224x224 yapalım (modelle uyumlu)
data = []
labels = []

# Yol düzeltildi — teeth_dataset/Training klasörüne göre
dataset_path = '../teeth_dataset/Training'  # Kendi klasör yolunu buraya ekle

# Veri setinden görselleri al
for label in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, label)
    if not os.path.isdir(class_path):
        continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)  # Görüntüyü RGB olarak oku
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Boyutlandır
        data.append(img)
        labels.append(0 if label.lower() == 'without_caries' else 1)

# Verileri numpy array formatına çevir
X = np.array(data) / 255.0  # Görüntüleri 0-1 arasında ölçekle
y = np.array(labels)

# 3. VERİYİ BÖL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. VERİ ARTIRMA (Data Augmentation)
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# 5. MODEL OLUŞTUR
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Yeni model katmanları ekle
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Çıkış katmanı
])

# Base modeli sabitle
base_model.trainable = False

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modelin özeti
model.summary()

# 6. MODELİ EĞİT
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

# Modeli eğit
model.fit(datagen.flow(X_train, y_train, batch_size=16), validation_data=(X_test, y_test), epochs=50, callbacks=[early_stopping, lr_scheduler])

# 7. MODELİ KAYDET
model.save('model/caries_model.h5')  # Modeli kaydet
print("✅ Model başarıyla kaydedildi!")