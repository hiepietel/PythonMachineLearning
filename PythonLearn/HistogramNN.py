import cv2
from pathlib import Path
import matplotlib.pyplot as plt
path = Path(".")

path = path.glob("../img/*.jpg")
import pandas as pd

images=[]
hists = []
y_train = pd.DataFrame()
x_train = pd.DataFrame()
for imagepath in path:

        img = cv2.imread(str(imagepath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (1920, 1080))
        images.append(img)
        print(imagepath)
        hist = pd.DataFrame(cv2.calcHist(img, [0], None, [256], [0, 256]))
        hists.append(hist)
        x_train.append(hist)
        a = pd.DataFrame([1])
        y_train.append(a)
#        plt.plot(hist)
#        plt.xlim([0,256])
#        plt.show()
print(len(images))
print(hist.shape)
hist_shape = (255)


X_train = pd.DataFrame(hist)
Y_train = pd.DataFrame(y_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPool1D, Flatten


model = Sequential()
model.add(Flatten())
model.add(Dense(32, input_shape=(1,256)))


model.add(Dense(2, activation='softmax'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, epochs=10)
print(model.summary())
