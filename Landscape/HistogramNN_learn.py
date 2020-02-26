from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

hist_path = [["hist.csv", 'desert_model.h5'],
             ["hist_center.csv", "desert_model_center.h5"]]

hist_number = 1



hists_df = pd.read_csv(hist_path[hist_number][0])

X = hists_df.drop('imagePath', axis=1).drop('isLandmark', axis=1).values
y = hists_df['isLandmark'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm

from tensorflow.keras.callbacks import TensorBoard
log_directory = 'logs\\fit'

board = TensorBoard(log_dir=log_directory,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch',
                    profile_batch=2,
                    embeddings_freq=1)


model = Sequential()

model.add(Dense(256,  activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128,  activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train,
          y=y_train,
          epochs=250,
          batch_size=256,
          validation_data=(X_test, y_test),
          callbacks=[board]
          )
from tensorflow.keras.models import load_model
model_path= '../db/Model/'+hist_path[hist_number][1]
print(model_path)
model.save('../db/Model/'+hist_path[hist_number][1])

metrics = pd.DataFrame(model.history.history)
metrics[['loss', 'val_loss']].plot()
plt.show()

print("exit")


