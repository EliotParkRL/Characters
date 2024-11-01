from PIL import Image
import numpy as np
import pandas as pd
# Load the image

image_path = 'characters/Img/img001-001.png'  # Update with your image path
image = Image.open(image_path)

# Convert the image to black and white (1-bit pixels)
bw_image = image.convert('1')  # '1' mode is for 1-bit pixels, black and white

# Convert the image to a NumPy array
binary_array = np.array(bw_image)

# Convert the boolean array (True for white, False for black) to binary (1s and 0s)
binary_array = binary_array.astype(int)

flattened_array = binary_array.flatten()

# Save to CSV
csv_path = 'images.csv'  # Define your CSV file path
pd.DataFrame([flattened_array]).to_csv(csv_path, index=False, header=False)