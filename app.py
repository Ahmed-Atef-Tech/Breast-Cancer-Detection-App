import os
import webbrowser
from flask import Flask, render_template, request, url_for, redirect
from PIL import Image, ImageOps
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import threading

app = Flask(__name__)

# Base path to your data
base_path = r"H:\Self learning Data\TIEC Innov Egypt\TIEC Final Project\Website\Flask version\AI APP Complete\AI APP"

# Function to train or load the model
def train_or_load_model(model_type):
    model_path = os.path.join(base_path, f"{model_type}_model.h5")
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"{model_type} model loaded from disk.")
    else:
        print(f"Training {model_type} model...")
        # Define the model architecture
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Set up data generators
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_generator = train_datagen.flow_from_directory(
            os.path.join(base_path, model_type, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            os.path.join(base_path, model_type, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )

        # Train the model
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // 32,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // 32,
            epochs=10
        )

        # Save the model
        model.save(model_path)
        print(f"{model_type} model trained and saved to disk.")

    return model

# Train or load models at startup
model1 = train_or_load_model("Mammogram")
model2 = train_or_load_model("Ultrasound")

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/upload')
def index():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('aboutpg.html')

@app.route('/team')
def team():
    return render_template('teamspg.html')

@app.route('/solutions')
def solutions():
    # Assuming solutions page is same as home page for now
    return render_template('Home.html')

@app.route('/pricing')
def pricing():
    # Placeholder for pricing page
    return render_template('pricing.html')

@app.route('/signup')
def signup():
    return render_template('signuppg.html')

@app.route('/login')
def login():
    return render_template('loginpg.html')

@app.route('/process', methods=['POST'])
def process_images():
    # Load the labels
    class_names1 = open(os.path.join(base_path, "Mammogram", "labels.txt"), "r").readlines()
    class_names2 = open(os.path.join(base_path, "Ultrasound", "labels.txt"), "r").readlines()

    # Get the uploaded files
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')

    results = []

    # Create uploads directory if it doesn't exist
    uploads_dir = os.path.join(app.static_folder, 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    if image1:
        # Process first image
        image = Image.open(image1).convert("RGB")
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = image_array.astype(np.float32) / 255.0
        data = np.expand_dims(normalized_image_array, axis=0)
        prediction = model1.predict(data)
        index = int(prediction[0][0] > 0.5)
        class_name = class_names1[index].strip()
        confidence_score = prediction[0][0] if index == 1 else 1 - prediction[0][0]

        # Generate description based on prediction
        if class_name.lower() == "benign":
            description = "The mammogram indicates benign findings. No signs of cancer detected."
            background = "linear-gradient(135deg, #143621 0%, #148344 100%)"  # Green background
        elif class_name.lower() == "malignant":
            description = "The mammogram indicates malignant findings. Further evaluation is recommended."
            background = "linear-gradient(135deg, #451212 0%, #a61f1f 100%)"  # Red background
        else:
            description = "The mammogram results are inconclusive. Please consult a specialist."
            background = "linear-gradient(135deg, #363514 0%, #836b14 100%)"  # Yellow background

        # Save uploaded image to static/uploads
        image1_filename = 'uploaded_mammogram.png'
        image.save(os.path.join(uploads_dir, image1_filename))

        results.append({
            'type': 'Mammogram',
            'class_name': class_name,
            'confidence_score': confidence_score,
            'description': description,
            'background': background,
            'image_filename': image1_filename
        })
    else:
        image1_filename = None

    if image2:
        # Process second image
        image = Image.open(image2).convert("RGB")
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = image_array.astype(np.float32) / 255.0
        data = np.expand_dims(normalized_image_array, axis=0)
        prediction = model2.predict(data)
        index = int(prediction[0][0] > 0.5)
        class_name = class_names2[index].strip()
        confidence_score = prediction[0][0] if index == 1 else 1 - prediction[0][0]

        # Generate description based on prediction
        if class_name.lower() == "benign":
            description = "The ultrasound indicates benign findings. No signs of cancer detected."
            background = "linear-gradient(135deg, #143621 0%, #148344 100%)"  # Green background
        elif class_name.lower() == "malignant":
            description = "The ultrasound indicates malignant findings. Further evaluation is recommended."
            background = "linear-gradient(135deg, #451212 0%, #a61f1f 100%)"  # Red background
        else:
            description = "The ultrasound results are inconclusive. Please consult a specialist."
            background = "linear-gradient(135deg, #363514 0%, #836b14 100%)"  # Yellow background

        # Save uploaded image to static/uploads
        image2_filename = 'uploaded_ultrasound.png'
        image.save(os.path.join(uploads_dir, image2_filename))

        results.append({
            'type': 'Ultrasound',
            'class_name': class_name,
            'confidence_score': confidence_score,
            'description': description,
            'background': background,
            'image_filename': image2_filename
        })
    else:
        image2_filename = None

    # Get patient's name and age
    full_name = request.form.get('fullName')
    age = request.form.get('age')

    return render_template('result.html', results=results, full_name=full_name, age=age)

if __name__ == "__main__":
    def open_browser():
        webbrowser.open_new('http://localhost:5000/')
    # Open the browser only once
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1, open_browser).start()
    app.run(debug=True)
