import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# 🔧 Yollar
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'teeth_dataset')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model', 'final_model.keras')

# 🔧 Parametreler
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 30

# 🔁 Veriyi oku
def load_data_recursive(dataset_path):
    data, labels, class_names = [], [], []
    for class_idx, disease_name in enumerate(sorted(os.listdir(dataset_path))):
        disease_path = os.path.join(dataset_path, disease_name)
        if not os.path.isdir(disease_path):
            continue
        class_names.append(disease_name)
        for root, _, files in os.walk(disease_path):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0
                    data.append(img)
                    labels.append(class_idx)
    return np.array(data), np.array(labels), class_names

print("📦 Veri yükleniyor...")
X, y, CLASS_NAMES = load_data_recursive(DATASET_PATH)
NUM_CLASSES = len(CLASS_NAMES)
print(f"✅ {len(X)} örnek yüklendi. Sınıflar: {CLASS_NAMES}")

# 🔀 Böl
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# 🔄 Augmentasyon
train_gen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator()

# 🧠 Model
def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    base_model.trainable = True  # Fine-tuning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

model = build_model()

# 🛑 Callbackler
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# 🚀 Eğit
history = model.fit(
    train_gen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    validation_data=test_gen.flow(X_test, y_test_cat),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# 🧪 Tahmin
y_probs = model.predict(X_test)
y_preds = np.argmax(y_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_preds, target_names=CLASS_NAMES))

# 📉 Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_preds), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
plt.close()

# 📈 ROC Curve
plt.figure(figsize=(10, 6))
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'roc_curve.png'))
plt.close()

# 📈 Eğitim Grafikleri
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
plt.close()

# 💾 Kaydet
model.save(MODEL_SAVE_PATH)
print("✅ Eğitim tamamlandı ve model kaydedildi.")