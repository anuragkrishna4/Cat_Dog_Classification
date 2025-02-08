from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

app = Flask(__name__)

# Load the trained model
model = load_model("C:\\Users\\KIIT\\OneDrive\\Desktop\\AD Lab\\cat_dog_classification\\catdog_model.h5")
classes = ["Cat", "Dog"]

# Check model input shape
print("Expected Model Input Shape:", model.input_shape)

def preprocess_image(image_path):
    """ Load and preprocess image to match model input """
    image = cv2.imread(image_path)               # Load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (256,256))        # Resize to (256,256)
    image = image.astype("float32") / 255.0      # Normalize
    image = image.reshape(1,256,256,3)  # Flatten the image (1, 150*150*3) if needed
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    try:
        img_path = "temp.jpg"
        file.save(img_path)

        # Preprocess image
        img_array = preprocess_image(img_path)

        # Predict
        prediction = model.predict(img_array)
        class_index = int(np.round(prediction[0][0]))  # Binary classification
        result = classes[class_index]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
