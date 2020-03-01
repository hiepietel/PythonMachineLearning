import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x = np.linspace(0, 50, 501)
y = np.sin(x)

plt.plot(x,y)
plt.show()

df = pd.DataFrame(data=y, index=x, columns=['Sine'])
print(len(df))

test_percent = 0.1
test_point = np.round(len(df)*test_percent)
test_ind = int(len(df)-test_point)

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

length = 50
batch_size = 1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)

print(len(scaled_train))

print(len(generator))

X, y = generator[0]
print(X,y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

n_features = 1

model = Sequential()

model.add(SimpleRNN(50, input_shape=(length, n_features)))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

print(model.summary())

model.fit_generator(generator, epochs=5)

losses = pd.DataFrame(model.history.history)

plt.plot(losses)
plt.show()

first_eval_batch = scaled_train[-length:]
first_eval_batch = first_eval_batch.reshape((1,length, n_features))

model.predict(first_eval_batch)

test_predictions = []
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

predicted_value = [[[99]]]
np.append(current_batch[:, 1:, :],[[[99]]], axis=1)

for i in range(len(test)):

    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)

    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

true_predictions = scaler.inverse_transform(test_predictions)

test['Predictions'] = true_predictions

plt.plot(test)
plt.show()

from tensorflow.keras.callbacks import EarlyStopping
early_stops = EarlyStopping(monitor="val_loss", patience=2)

length = 49
generator = TimeseriesGenerator(scaled_train, scaled_train, length=len, batch_size=1)

validation_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=1)