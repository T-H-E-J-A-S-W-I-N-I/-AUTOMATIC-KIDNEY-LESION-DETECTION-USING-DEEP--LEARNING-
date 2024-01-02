import os

import cv2
import numpy as np
from PIL import Image

import tensorflow as tf

# Flask utils
from flask import Flask, render_template, request


# Define a flask app

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
model = tf.saved_model.load('saved_model')
print(list(model.signatures.keys()))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def predict_single_img(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Load the SavedModel
    model = tf.saved_model.load('saved_model')
    prediction_fn = model.signatures['serving_default']

    # Make the prediction
    output = prediction_fn(tf.constant(img, dtype=tf.float32))
    print(output)
    predictions = output['dense_3']

    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class


print(predict_single_img("static/uploads/Cyst- (7).jpg"))


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No image uploaded.')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No image uploaded.')

        if not allowed_file(file.filename):
            return render_template('index.html', prediction='Invalid file type.')

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        prediction = predict_single_img(filename)

        labels = ['Cyst', 'Normal', 'Stone', 'Tumor']  # Update with your class labels

        result = labels[prediction]

        print(result)

        return result

    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


if __name__ == '__main__':
    app.run()
