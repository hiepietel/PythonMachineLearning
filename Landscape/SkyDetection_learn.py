from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.models import Sequential

import matplotlib.pylab as plt
from matplotlib.image import imread

import numpy as np
import cv2
from pathlib import Path

batch_size = 128 * 3
num_classes = 11
epochs = 60

# input image dimensions
img_x, img_y = 32, 32

input_shape = (32, 32, 3)

def load_data(path):
    images = np.array([cv2.imread(str(img)) for img in path])
    return images

def return_path(main_folder, folder):
    path = Path(".")
    return path.glob('../db/img/' + main_folder + '/' + folder + '/*.jpg')


main_folder = "sky_correct"

path = return_path(main_folder, 'train_true')
x_train = load_data(path)

path = return_path(main_folder, 'train_false')
y_train = load_data(path)

path = return_path(main_folder, 'test_true')
x_test = load_data(path)

path = return_path(main_folder, 'test_false')
y_test = load_data(path)


# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)
# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes=None)
y_test = keras.utils.to_categorical(y_test, num_classes=None)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

results = model.fit_generator(x_train, y_train, epochs=20,
                             validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
