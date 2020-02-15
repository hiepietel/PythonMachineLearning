import os

data_dir = "..\\cell_images"

print(os.listdir(data_dir))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.image import imread

test_path = data_dir + '\\test'
train_path = data_dir + '\\train'

print(os.listdir(test_path))
print(os.listdir(train_path))

train_path_parasitized = train_path+'\\parasitized' + '\\'
train_path_uninfected = train_path+'\\uninfected' + '\\'

test_path_parasitized = test_path + '\\parasitized' + '\\'
test_path_uninfected = test_path + '\\uninfected' + '\\'

print(os.listdir(train_path_parasitized))
print(os.listdir(train_path_uninfected))

print(len(os.listdir(train_path_parasitized)))
print(len(os.listdir(train_path_uninfected)))

dim1 = []
dim2 = []
un  = None
for image_file_name in os.listdir(test_path_uninfected):
    img = imread(test_path_uninfected+image_file_name)
    d1, d2, colors = img.shape
    dim1.append((d1))
    dim2.append((d2))
    un = img

sns.jointplot(dim1, dim2)
plt.show()

print(np.mean(dim1))
print(np.mean(dim2))

image_shape= (130, 130, 3)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               #rescale = 1 / 255
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest')

plt.imshow(un)
plt.show()

plt.imshow(image_gen.random_transform(un))
plt.show()

print(image_gen.flow_from_directory(train_path))
print(image_gen.flow_from_directory(test_path))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)

batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                class_mode='binary',
                                               shuffle=False)

print(train_image_gen.class_indices)
#results = model.fit_generator(train_image_gen, epochs=20,
#                              validation_data=test_image_gen,
#                              callbacks=[early_stop])

from tensorflow.keras.models import load_model

model = load_model('../db/Model/malaria_detector.h5')

print(model.summary())

model.evaluate(test_image_gen)

#evaluating the model

pred = model.predict_generator(test_image_gen)

print(pred)

predictions = pred > 0.80

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(test_image_gen.classes, predictions))
print(confusion_matrix(test_image_gen.classes, predictions))