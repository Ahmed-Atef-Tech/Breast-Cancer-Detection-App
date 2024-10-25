# Breast Cancer Detection Application

This project is a Breast Cancer Detection Application, developed in Python, using **Flask** for the web interface and **Tkinter** for the desktop GUI. The application can analyze mammogram and ultrasound images to detect cancerous signs, providing results with a confidence score. The detection models are based on convolutional neural networks (CNNs) and use TensorFlow/Keras for implementation.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Flask Web App](#running-the-flask-web-app)
  - [Running the Tkinter Desktop App](#running-the-tkinter-desktop-app)
- [Project Structure](#project-structure)
- [Models](#models)
- [License](#license)

## Features

- **Web Interface**: Provides a web-based interface for users to upload images and view results.
- **Desktop Interface**: Users can upload images and view results in a Tkinter-based GUI.
- **Image Mirroring Tool**: Includes an image mirroring script to preprocess images.
- **Deep Learning Model**: CNN models for mammogram and ultrasound image classification, using binary classification to identify benign and malignant findings.

## Requirements

- Python 3.6+
- TensorFlow 2.0+
- Flask
- Tkinter
- Pillow (PIL)
- tqdm

## Usage
![24_10_25_23_49_00](https://github.com/user-attachments/assets/c7fb0563-86b6-419c-81d0-6e1c827d1e33)
--------------------------------------------------------
![24_10_25_23_49_16](https://github.com/user-attachments/assets/8dddaeed-78e1-4b59-98f5-3f226a46e272)
---
![24_10_25_23_48_54](https://github.com/user-attachments/assets/86634c26-7af1-4e62-96a2-5efd115eb25c)


### Running the Flask Web App

1. **Start the Flask server**:
   ```bash
   python app.py
   ```
   The app will automatically open in your default browser at `http://localhost:5000`.

2. **Using the Web Interface**:
   - Upload mammogram and ultrasound images via the provided forms.
   - View the results, including predicted class (benign or malignant) and confidence score.

### Running the Tkinter Desktop App

1. **Launch the Desktop App**:
   ```bash
   python Application.py
   ```
2. **Using the Desktop Interface**:
   - Use the "Upload" buttons to upload mammogram and ultrasound images.
   - Click "Process Images" to get predictions displayed on the interface.

## Project Structure

- `app.py`: Flask web application code handling image uploads, predictions, and web routes.
- `main.py`: Image preprocessing tool using PIL for mirroring images.
- `Application.py`: Tkinter GUI application for desktop-based image upload and classification.
- `templates/`: HTML templates for the Flask web application.

## Models

The application trains two convolutional neural network (CNN) models for mammogram and ultrasound image classification. The models are structured as follows:

1. **Input Layer**: 224x224 image input.
2. **Convolutional Layers**: Three Conv2D layers with max-pooling.
3. **Flatten and Dense Layers**: Flattened layer followed by dense layers with a final sigmoid output.
