import cv2
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

def returnColumns(end):
        col = []
        for i in range(end):
                col.append("img:"+str(i))
        return col
name = 'desert'
path = Path(".")
path = path.glob('../img/'+name+'/*.jpg')

hists = []
hists_df = pd.DataFrame()

images_df = pd.DataFrame()
for imagepath in path:

        img = cv2.imread(str(imagepath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (1920, 1080))

        print(imagepath)
        hist = cv2.calcHist(img, [0], None, [256], [0, 256])

        hist_df = pd.DataFrame(hist.reshape(-1, len(hist)), columns=returnColumns(len(hist)))
        hists_df = hists_df.append(hist_df,ignore_index = True)


hists_df.plot(kind='line', title='desert', legend=False)

plt.show()
hists_df.to_csv(r''+name+'hist.csv',header=True,index=False)


# X_train = pd.DataFrame(hist)
# Y_train = pd.DataFrame(y_train)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPool1D, Flatten
#
#
# model = Sequential()
# model.add(Flatten())
# model.add(Dense(32, input_shape=(1,256)))
#
#
# model.add(Dense(2, activation='softmax'))
#
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X_train, epochs=10)
# print(model.summary())
