from PIL import Image
import numpy as np

# Load the image
image_path = 'path/to/your/image.png'  # Update with your image path
image = Image.open(image_path)

# Convert the image to black and white (1-bit pixels)
bw_image = image.convert('1')  # '1' mode is for 1-bit pixels, black and white

# Convert the image to a NumPy array
binary_array = np.array(bw_image)

# Convert the boolean array (True for white, False for black) to binary (1s and 0s)
binary_array = binary_array.astype(int)

# Display the binary array
print(binary_array)