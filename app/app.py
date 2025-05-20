from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import cv2

app = Flask(__name__)

# Modeli yükle
model = load_model('../model/caries_model.h5')
classes = ['No Caries', 'Caries']  # 0=No Caries, 1=Caries

# Görseli ön işleme (eğitimle aynı işlemler)
def preprocess_image(file_path, target_size=(224, 224)):
    # Görseli OpenCV ile oku ve eğitimdeki gibi işle
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("Görsel okunamadı veya bozuk")
    
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Eğitimdeki gibi RGB formatına dönüştür
    img_array = np.array(img) / 255.0  # 0-1 arası normalleştir
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi.'})

    try:
        # Dosyayı geçici olarak kaydet
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Görüntüyü işle
        img_array = preprocess_image(file_path)

        # Tahmin yap
        prediction_prob = model.predict(img_array)[0][0]  # Sigmoid çıkışı
        predicted_class = classes[1] if prediction_prob >= 0.5 else classes[0]
        confidence = prediction_prob if predicted_class == classes[1] else 1 - prediction_prob

        # Sonuçları yuvarla
        confidence = round(float(confidence), 4)

        # Temizlik
        os.remove(file_path)

        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'probability': float(prediction_prob)  # Ham olasılık değeri
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)