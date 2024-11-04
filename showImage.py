import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

image_data = np.array(pd.read_csv("output.csv").iloc[2])
image_2d = image_data.reshape((32, 32))

# Display the image
plt.imshow(image_2d, cmap='gray')
plt.axis('off')  # Hide the axes for clarity
plt.show()