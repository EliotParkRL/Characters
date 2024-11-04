import os
import numpy as np
import pandas as pd
from PIL import Image
from natsort import natsorted


def pngs_to_csv(folder_path, output_csv, resize_dim=(32, 32)):
    """
    Convert all PNG images in a folder to a CSV where each row represents an image's pixel values.
    Images are first scaled to a specified resolution.

    :param folder_path: Path to the folder containing PNG images.
    :param output_csv: Name of the output CSV file.
    :param resize_dim: Tuple specifying the (width, height) for resizing each image.
    """
    # Initialize an empty list to hold all flattened images
    data = []

    # Iterate through each PNG file in the folder
    for filename in natsorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            # Open the image, convert to grayscale, resize, and flatten
            img = Image.open(os.path.join(folder_path, filename)).convert('L')
            img = img.resize(resize_dim)  # Resize to specified dimensions

            # Flatten the resized image to a 1D array of pixel values
            img_array = np.array(img).flatten()

            # Append the flattened image to the data list
            data.append(img_array)

    # Convert the list to a DataFrame, each row is an image
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)


# Usage
pngs_to_csv('/Users/natha/Documents/GitHub/Characters/characters/Img', 'output2.csv', resize_dim=(64, 64))
