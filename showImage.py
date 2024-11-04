import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

image_data = np.array(pd.read_csv("output.csv").iloc[1])
image_2d = image_data.reshape((64, 64))

# Display the image
plt.imshow(image_2d, cmap='gray')
plt.axis('off')  # Hide the axes for clarity
plt.show()