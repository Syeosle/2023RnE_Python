import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X = np.array([[2.24, 3.29, 0],[3.28,2,9.3],[1,5.3,8.2],[3.333,2.24,3]])
y = np.array([1012,5432,243,4324])

model = Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))

sgd = optimizers.SGD(learning_rate=10**(-5))
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(X, y, epochs=2048)

print(model.predict(X))