from PIL import Image
import os
import numpy as np
import csv

# Directory containing the grayscale PNG images
directory_path = "/Users/eliotpark/Downloads/characters/Img"

# Threshold value for binary conversion
threshold_value = 128

# Output CSV file path
output_csv_path = "binary_vectors.csv"

# Prepare data for CSV writing
csv_data = []

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".png"):
        # Load the image, convert to grayscale, and resize to 32x32 pixels
        image_path = os.path.join(directory_path, filename)
        image = Image.open(image_path).convert("L").resize((32, 32))  # Convert to grayscale and resize to 32x32
        
        # Convert image to numpy array
        image_array = np.array(image)
        
        # Apply thresholding to convert to binary (0s and 1s)
        binary_image = (image_array >= threshold_value).astype(int)
        
        # Flatten the binary image to a 1D vector
        binary_vector = binary_image.flatten()
        
        # Append filename and vector as a row in the CSV data
        csv_data.append([filename] + binary_vector.tolist())
        print(filename)

# Write the data to a CSV file
with open(output_csv_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)

print(f"Binary vectors have been saved to {output_csv_path}")
