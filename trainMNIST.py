import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf

# Suppress TensorFlow logging for clarity
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Load the MNIST dataset
(X, y), (X_test, y_test) = mnist.load_data()

# Normalize the input data
X = X.reshape(-1, 28 * 28) / 255.0
X_test = X_test.reshape(-1, 28 * 28) / 255.0

# Split the training data into training and validation sets
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2, random_state=42)

# Set random seed for reproducibility
tf.random.set_seed(1234)

# Define the model
model = Sequential(
    [
        Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    ],
    name="MNIST_Classifier"
)

# Compile the model
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=False),
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

# Function to calculate categorization error
def eval_cat_err(y, yhat):
    """
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)
    """
    m = len(y)
    incorrect = np.sum(yhat != y)
    err = incorrect / m
    return err

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_cv, y_cv))

# Predict function
model_predict = lambda Xl: np.argmax(model.predict(Xl), axis=1)

# Evaluate categorization error on training, validation, and test sets
training_cerr = eval_cat_err(y_train, model_predict(X_train))
cv_cerr = eval_cat_err(y_cv, model_predict(X_cv))
test_cerr = eval_cat_err(y_test, model_predict(X_test))

print(f"Categorization error, training: {training_cerr:.3f}")
print(f"Categorization error, validation: {cv_cerr:.3f}")
print(f"Categorization error, test: {test_cerr:.3f}")
