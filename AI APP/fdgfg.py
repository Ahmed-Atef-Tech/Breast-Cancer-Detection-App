import os
import csv
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Base path
base_path = r"H:\Self learning Data\TIEC Innov Egypt\TIEC Final Project\Website\Flask version\AI APP Complete\AI APP"

# Load the trained model
model_path = os.path.join(base_path, 'Mammogram', 'trained_model.h5')
model = load_model(model_path)

# Path to the folder containing images for prediction
image_folder = os.path.join(base_path, 'Mammogram', 'test', 'Density1Benign')

# Output CSV file to store results
output_csv = os.path.join(base_path, 'predictions.csv')

# Initialize a list to store results
results = []

# Iterate through images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, filename)

        # Load and preprocess the image for prediction
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Make predictions
        prediction = model.predict(img_array)
        result = {'Filename': filename, 'Prediction': prediction[0][0]}
        results.append(result)

# Save results to a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['Filename', 'Prediction']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Predictions saved to {output_csv}")
