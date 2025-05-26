from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

# Logger ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Yapılandırmalar
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

CLASS_NAMES = [
    'Diş Taşı',              # Calculus
    'Çürük ve Diş Eti',      # Caries_Gingivitis
    'Diş Çürüğü',            # Data caries
    'Diş Eti İltihabı',      # Gingivitis
    'Diş Eksikliği',         # hypodontia
    'Ağız Yarası',           # Mouth Ulcer
    'Diş Renklenmesi'        # Tooth Discoloration
]

# 🔍 Dosya uzantısı kontrolü
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 📥 Modeli yükle (Keras formatı kullanılıyor)
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'final_model.keras')
    model = load_model(model_path)
    logger.info("✅ Model başarıyla yüklendi.")
except Exception as e:
    logger.error(f"❌ Model yüklenemedi: {str(e)}")
    model = None

# 🧼 Görseli işle
def preprocess_image(img_path, target_size=(160, 160)):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Görsel okunamadı.")

        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CLAHE ile histogram eşitleme
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logger.error(f"📛 Ön işleme hatası: {str(e)}")
        raise

# 🔮 Tahmin endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model yüklenemedi.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Görsel dosyası yüklenmedi.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Geçersiz dosya tipi. PNG, JPG, JPEG desteklenir.'}), 400

    try:
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # Temizlik: yüklenen dosyayı sil
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Geçici dosya silinemedi: {str(e)}")

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence, 4),
            'all_probabilities': {
                CLASS_NAMES[i]: round(float(prob), 4) for i, prob in enumerate(predictions)
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 🚀 Sunucu başlat
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)