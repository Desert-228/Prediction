from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Load your pre-trained model
loaded_model = tf.keras.models.load_model('C:\\Users\\malya\\OneDrive\\Desktop\\Soil\\vgg19.h5')

# Load crop data
crop_data = pd.read_csv("C:\\Users\\malya\\OneDrive\\Desktop\\Soil\\crop_Agri.csv")

# Get class labels
class_labels = os.listdir('C:\\Users\\malya\\OneDrive\\Desktop\\Soil\\Soil types')

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for processing the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sample_image_path = request.form['image']
        sample_image = Image.open(sample_image_path).resize((224, 224))
        sample_image_array = np.expand_dims(np.array(sample_image), axis=0) / 255.0
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])

        # Generate predictions for the input image
        predictions = loaded_model.predict(sample_image_array)
        soil_class = np.argmax(predictions)

        # Get soil class label
        soil_class_label = class_labels[soil_class]

        # Print prediction result for the input image
        print(f"Predicted class: {soil_class_label}")

        # Use the predicted soil class to get crop predictions
        input_data = pd.DataFrame({
            'Temperature': [temp],
            'SoilType': [soil_class_label],
            'Temperature_and_Rainfall': [rain]
        })

        # Perform one-hot encoding
        input_data_encoded = pd.get_dummies(input_data)

        # Make predictions for crops
        crop_predictions = loaded_model.predict(input_data_encoded)
        crops = crop_predictions[0]

        return render_template('result.html', soil_class=soil_class_label, crop_predictions=crops)

if __name__ == '__main__':
    app.run(debug=True)
