import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

X = np.array(pd.read_csv("output.csv"))
y= pd.read_csv("processed_labels.csv").iloc[:,1]
y = pd.factorize(y)[0]
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)

tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(62, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    ], name="Complex"
)
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=False),
    optimizer=Adam(learning_rate=0.0001),
)
def eval_cat_err(y, yhat):
    """
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      err: (scalar)
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
        if yhat[i] != y[i]:
            incorrect += 1
    err = incorrect/m
    return(err)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_cv, y_cv),
    callbacks=[early_stopping],
    shuffle=True
)

model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)

training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}")