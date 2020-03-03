import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('../db/Model/sky.h5')

img_path = '../db/img/desert_sand_man_s.jpg'

image = cv2.imread(img_path)
image_rect = np.copy(image)
image_new = np.copy(image)
i = 0
j = 0
finished = False
count_x = 0
xount_y = 0

width = image.shape[0]
height = image.shape[1]
sky_percent = 0.3
debug = 1
size = 32

# while not finished:

for i in range(0, width-size, size):
    for j in range(0, height-size, size):

        sky_piksels = 0
        next_i = i + size
        next_j = j + size
        # for i in range(i, next_i + 1):
        #     for j in range(j, next_j + 1):

        # little_sky = np.array((32, 32, 3))
        little_sky = image[i:i + size, j:j + size]
        if little_sky.shape == (32, 32, 3):
            my_little_sky = np.expand_dims(little_sky, axis=0)
            # my_little_sky = image.img_to_array(my_little_sky)
            #print(type(my_little_sky))
            #print(my_little_sky.shape)

            value = model.predict(my_little_sky)
            #print(value)
            if value[0][0] < 0.5:

                cv2.rectangle(image_rect, (j, i), (j + size, i + size), (0, 200, 0), -1)
                  # Transparency factor.

                # Following line overlays transparent rectangle over the image

    # if i + 2 * size < width:
    #     i += size
    # else:
    #     if j + 2 * size < height:
    #         j += size
    #     else:
    #         finished = True

alpha = 0.2
cv2.addWeighted(image_rect, alpha, image, 1 - alpha, 0, image_rect)
cv2.imshow("res", image_rect)
cv2.waitKey(0)