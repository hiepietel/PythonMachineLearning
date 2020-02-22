import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)
single_image = x_train[0]
plt.imshow(single_image)
plt.show()

print(y_train)

from tensorflow.keras.utils import to_categorical

print(y_train.shape)
y_example = to_categorical(y_train)
print(y_train.shape)
print(y_example[0])
y_cat_test = to_categorical(y_test, num_classes=10)
y_cat_train = to_categorical(y_train, num_classes=10)
print(y_cat_train)
print(y_cat_test)

print(single_image.max())
print(single_image.sum())
print(single_image.min())

x_train = x_train / 255
y_test = y_test / 255

#batch_size, width, height, color_channels
x_train = x_train.reshape(60000, 28, 28, 1)

x_test = x_test.reshape(10000,28, 28, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
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

model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu')) #convolutional layer
model.add(MaxPool2D(pool_size=(2, 2))) #pooling layer

model.add(Flatten())

model.add(Dense(128, activation='relu'))

#Outputlayer  soft max -> multi class

model.add(Dense(10, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=1)
with tf.device("/device:GPU:0"):
    model.fit(x_train, y_cat_train, epochs=10, validation_data=(x_test, y_cat_test), callbacks=[early_stop, board])

model.save('../db/Model/mnist_model.h5')

metrics = pd.DataFrame(model.history.history)
metrics[['loss', 'val_loss']].plot()

plt.show()
metrics[['accuracy', 'val_accuracy']].plot()

plt.show()


model.evaluate(x_test, y_cat_test, verbose=0)

from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict_classes(x_test)

print(y_cat_test.shape)

#print(classification_report(y_test, predictions))

#confusion_matrix(y_test, predictions)

# import seaborn as sns
#
# plt.figure(figsize=(10, 6))
# sns.heatmap(confusion_matrix(y_test, predictions))
# plt.show()


