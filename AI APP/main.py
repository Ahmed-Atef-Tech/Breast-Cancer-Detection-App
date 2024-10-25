from PIL import Image
from tqdm import tqdm
import os

def mirror_image(input_path, output_path):
    # Open the image file
    original_image = Image.open(input_path)

    # Mirror the image
    mirrored_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Save the mirrored image
    mirrored_image.save(output_path)

if __name__ == "__main__":
    # Specify the directory where your images are stored
    base_path = r"H:\Self learning Data\TIEC Innov Egypt\TIEC Final Project\Website\Flask version\AI APP Complete\AI APP"
    input_directory = os.path.join(base_path, "Images", "5", "b5")

    # Specify the directory where you want to save mirrored images
    output_directory = os.path.join(base_path, "Preprocces_Images", "5")

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get the list of files to process
    files_to_process = [filename for filename in os.listdir(input_directory) if filename.endswith("R.jpg")]

    # Create a progress bar
    progress_bar = tqdm(total=len(files_to_process), desc="Converting images")

    # Loop through files in the input directory
    for filename in files_to_process:
        input_path = os.path.join(input_directory, filename)

        # Modify the output filename as needed
        output_filename = filename[:-5] + "_mirrored.jpg"
        output_path = os.path.join(output_directory, output_filename)

        # Mirror the image and save it
        mirror_image(input_path, output_path)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()
