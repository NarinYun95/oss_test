from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Keras 모델 불러오기
model = tf.keras.models.load_model('/converted_keras/keras.model.h5')

def preprocess_image(image):
    # 모델이 요구하는 224x224 크기로 이미지 크기 조정
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    # 이미지를 numpy 배열로 변환하고 정규화
    image_array = np.asarray(image) / 255.0
    # 배치 차원 추가
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('Image_test.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return '파일이 업로드되지 않았습니다.', 400
    file = request.files['file']
    image = Image.open(file.stream)
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    # 이진 분류를 가정하고, 모델에 따라 인덱스를 조정
    result = '자폐증 감지' if predictions[0][0] > 0.5 else '비자폐증'
    return result

if __name__ == '__main__':
    app.run(debug=True)
