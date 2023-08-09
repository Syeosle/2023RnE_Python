import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

data = pd.read_csv('LaptopData.csv', encoding='utf-8')

for column in ('Brand', 'CpuMf', 'GpuMf') :
    unique = data[column].unique()
    data[column] = data[column].replace(unique, [i for i in range(len(unique))])

data_X = data[data.columns[1:16].tolist()].values
data_Y = data[data.columns[16]].values

print(data_X.dtype)
print(data_Y.dtype)

model = Sequential()
model.add(Dense(1000, input_dim=15, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(1, activation='linear'))

sgd = optimizers.SGD(learning_rate=10**(-5))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(data_X, data_Y, epochs=16384)

X1 = np.array([[0, ]])

print(model.predict(data_X))