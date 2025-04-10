from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Modeli yükle
model = load_model('../model/caries_model.h5')  # Modelin yolu
classes = ['No Caries', 'Caries']  # Sınıflar

# Görseli ön işleme
def preprocess_image(file_path, target_size=(224, 224)):
    img = Image.open(file_path).convert('RGB')  # Görseli RGB formatına dönüştür
    img = img.resize(target_size)  # Boyutlandır
    img_array = np.array(img) / 255.0  # 0-1 arası normalleştir
    img_array = np.expand_dims(img_array, axis=0)  # Modelin input formatı
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi.'})

    files = request.files.getlist('file')  # Birden fazla dosya alıyoruz
    predictions = []

    for file in files:
        if file.filename == '':
            predictions.append({'error': 'Dosya seçilmedi.'})
            continue

        # Dosyayı static klasörüne kaydet
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Görüntüyü işle
        img_array = preprocess_image(file_path)

        # Tahmin yap
        prediction = model.predict(img_array)[0]
        predicted_class = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        # Görseli kaydet
        img = Image.open(file_path)
        img.save(f"static/{file.filename}")  # Resmi static klasörüne kaydet

        # Dosyayı kaydettikten sonra sil (isteğe bağlı olarak temizleme)
        os.remove(file_path)

        predictions.append({
            'class': predicted_class,
            'confidence': confidence
        })

    return jsonify(predictions)  # Birden fazla dosya için tahmin sonuçlarını döneriz

if __name__ == '__main__':
    app.run(debug=True)