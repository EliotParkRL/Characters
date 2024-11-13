import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split

# Load data
X = np.array(pd.read_csv("output.csv"))
y = np.array(pd.read_csv("processed_labels.csv").iloc[:, 1])

# Split data
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.20, random_state=1)

# Define model
tf.random.set_seed(1234)
model = Sequential([
    Dense(4096, activation="relu"),
    Dense(120, activation="relu"),
    Dense(40, activation="relu"),
    Dense(6, activation="linear")
])
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))

# Define the original evaluation function
def eval_cat_err(y, yhat):
    """
    Calculate the categorization error using the original method.
    Args:
      y    : (ndarray, Shape (m,))  target value of each example
      yhat : (ndarray, Shape (m,))  predicted value of each example
    Returns:
      err: (scalar) categorization error rate
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
        if yhat[i] != y[i]:
            incorrect += 1
    err = incorrect / m
    return err

# Unique classes (for reference)
classes = np.unique(y)

# Model prediction function
model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)), axis=1)

# Evaluate categorization error on training and cross-validation sets
training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))

print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}")