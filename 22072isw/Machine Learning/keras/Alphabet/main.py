import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv('AlphabetData.csv', encoding='utf-8')

X = data[data.columns[1:].tolist()].values
y = data[data.columns[0]].values
y = to_categorical(y)

(X, X1, y, y1) = train_test_split(X, y, train_size=0.7, random_state=1)

model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(26, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=200, batch_size=16, validation_data=(X1, y1))

epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()