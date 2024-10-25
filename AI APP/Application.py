import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

class BreastCancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer App")
        # Variables to store uploaded images
        self.image1_path = tk.StringVar()
        self.image2_path = tk.StringVar()

        # Create labels for image upload
        tk.Label(root, text="Mammogram:").grid(row=0, column=0, padx=10, pady=10)
        tk.Label(root, text="Ultra Sound:").grid(row=1, column=0, padx=10, pady=10)

        # Create image preview widgets
        self.image1_preview = tk.Label(root)
        self.image1_preview.grid(row=0, column=1, padx=10, pady=10)

        self.image2_preview = tk.Label(root)
        self.image2_preview.grid(row=1, column=1, padx=10, pady=10)

        # Create buttons to upload images
        tk.Button(root, text="Upload", command=lambda: self.upload_image(1)).grid(row=0, column=2, padx=10, pady=10)
        tk.Button(root, text="Upload", command=lambda: self.upload_image(2)).grid(row=1, column=2, padx=10, pady=10)

        # Create a button to process images
        tk.Button(root, text="Process Images", command=self.process_images).grid(row=2, column=1, pady=20)

        # Train models
        self.base_path = r"H:\Self learning Data\TIEC Innov Egypt\TIEC Final Project\Website\Flask version\AI APP Complete\AI APP"
        self.model1 = self.train_model("Mammogram")
        self.model2 = self.train_model("Ultrasound")

    def train_model(self, model_type):
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
            os.path.join(self.base_path, model_type, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.base_path, model_type, 'train'),
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

        return model

    def upload_image(self, image_number):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            image = Image.open(file_path)
            image = image.resize((150, 150))
            photo = ImageTk.PhotoImage(image)

            if image_number == 1:
                self.image1_path.set(file_path)
                self.image1_preview.config(image=photo)
                self.image1_preview.image = photo
            elif image_number == 2:
                self.image2_path.set(file_path)
                self.image2_preview.config(image=photo)
                self.image2_preview.image = photo

    def process_images(self):
        # Load the labels
        class_names1 = open(os.path.join(self.base_path, "Mammogram", "labels.txt"), "r").readlines()
        class_names2 = open(os.path.join(self.base_path, "Ultrasound", "labels.txt"), "r").readlines()

        # Create the array of the right shape to feed into the keras model
        data1 = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data2 = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Get the image paths from the uploaded photos
        image_path1 = self.image1_path.get()
        image_path2 = self.image2_path.get()

        if image_path1:
            # Process first image
            image1 = Image.open(image_path1).convert("RGB")
            size = (224, 224)
            image1 = ImageOps.fit(image1, size, Image.Resampling.LANCZOS)
            image_array1 = np.asarray(image1)
            normalized_image_array1 = (image_array1.astype(np.float32) / 255.0)
            data1[0] = normalized_image_array1
            prediction1 = self.model1.predict(data1)
            index1 = int(prediction1[0][0] > 0.5)
            class_name1 = class_names1[index1]
            confidence_score1 = prediction1[0][0] if index1 == 1 else 1 - prediction1[0][0]

            # Display result for the first image on the screen
            result_text1 = f"Class: {class_name1.strip()}   Confidence Score: {confidence_score1:.4f}"
            result_label1 = tk.Label(self.root, text=result_text1, font=('Helvetica', 14))
            result_label1.grid(row=3, column=0, columnspan=3, pady=10)

        if image_path2:
            # Process second image
            image2 = Image.open(image_path2).convert("RGB")
            size = (224, 224)
            image2 = ImageOps.fit(image2, size, Image.Resampling.LANCZOS)
            image_array2 = np.asarray(image2)
            normalized_image_array2 = (image_array2.astype(np.float32) / 255.0)
            data2[0] = normalized_image_array2
            prediction2 = self.model2.predict(data2)
            index2 = int(prediction2[0][0] > 0.5)
            class_name2 = class_names2[index2]
            confidence_score2 = prediction2[0][0] if index2 == 1 else 1 - prediction2[0][0]

            # Display result for the second image on the screen
            result_text2 = f"Class: {class_name2.strip()}   Confidence Score: {confidence_score2:.4f}"
            result_label2 = tk.Label(self.root, text=result_text2, font=('Helvetica', 14))
            result_label2.grid(row=4, column=0, columnspan=3, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = BreastCancerApp(root)
    root.mainloop()
