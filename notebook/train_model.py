import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# 📁 Yol Ayarları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'teeth_dataset')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model', 'final_model.keras')

# 🔧 Parametreler
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 30

# 🔁 Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
CLASS_NAMES = list(train_generator.class_indices.keys())
print(f"✅ Sınıflar: {CLASS_NAMES}")

# ⚖️ Class Weight Hesapla
y_train_labels = train_generator.classes
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
cw_dict = dict(enumerate(cw))

# 🧠 Model Oluştur
def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    base_model.trainable = True
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

# 📉 Callback'ler
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# 🚀 Eğit
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1
)

# 🧪 Tahmin ve Değerlendirme
y_probs = model.predict(val_generator, verbose=1)
y_preds = np.argmax(y_probs, axis=1)
y_true = val_generator.classes

print("\n📊 Classification Report:")
print(classification_report(y_true, y_preds, target_names=CLASS_NAMES))

# 🔷 Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_preds), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
plt.close()

# 📈 ROC Curve
y_true_cat = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)
plt.figure(figsize=(10, 6))
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'roc_curve.png'))
plt.close()

# 📉 Eğitim Grafikleri
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