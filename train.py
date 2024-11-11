import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.activations import relu,linear
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.python.keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


'''
X_train = pd.read_csv("output.csv")
y_train = pd.read_csv("labels.csv")[1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=1)
'''