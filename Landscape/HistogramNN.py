import cv2
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

def returnColumns(end):
        col = []

        for i in range(end):
                col.append("img"+str(i))
        return col

def returnHistogram(imagepath):
    img = cv2.imread(str(imagepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (1920, 1080))

    #print(imagepath)
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])

    hist_df = pd.DataFrame(hist.reshape(-1, len(hist)), columns=returnColumns(len(hist)))

    return hist_df

name = 'desert'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

hists_df = pd.DataFrame()

for imagepath in path:
        firstCol = pd.DataFrame(data={'imagePath': [imagepath], 'isLandmark': [1]})
        hist_df = pd.DataFrame()
        hist_df = pd.concat([firstCol, returnHistogram(imagepath)], axis=1)
        hists_df = hists_df.append(hist_df,ignore_index = True, sort=False)

name = 'fake'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

for imagepath in path:
        firstCol = pd.DataFrame(data={'imagePath': [imagepath], 'isLandmark': [0]})
        #hist_df = pd.DataFrame()
        hist_df = pd.concat([firstCol, returnHistogram(imagepath)], axis=1)
        hists_df = hists_df.append(hist_df,ignore_index = True, sort=False)


landscape_df =  hists_df.where(hists_df['isLandmark'] == 1).dropna().drop('imagePath', axis=1).drop('isLandmark', axis=1)
for i in range(len(landscape_df.index)):
    landscape_df.xs(i).plot()

plt.show(title='desert')

fake_df = hists_df.where(hists_df['isLandmark'] == 0).dropna().drop('imagePath', axis=1).drop('isLandmark', axis=1)
for i in range(len(fake_df.index)):
    fake_df.xs(i).plot()

plt.show()

#  hists_df.where(hists_df['isLandmark'] == 0).plot(kind='line', title='fake', legend=False)
#  plt.show()

hists_df.to_csv(r''+name+'hist.csv',header=True,index=False)

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

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train,
          y=y_train,
          epochs=250,
          batch_size=256,
          validation_data=(X_test, y_test),
          callbacks=[board]
          )
from tensorflow.keras.models import load_model

model.save('../db/Model/desert_model.h5')

print("exit")


