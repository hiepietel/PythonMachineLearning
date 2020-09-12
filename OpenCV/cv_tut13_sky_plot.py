import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#img = np.zeros((200,200), np.uint8)
#img = cv.imread('../db/img/desert/algodones_dunes_california-wallpaper-1920x1080.jpg', 1)
#img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

scaler = 1

height = 1080 / scaler
width = 1920 / scaler

height = int(height)
width = int(width)

img = cv.imread('../db/img/sky_best_not_converted/20170629_224036_HDR.jpg', 1)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (width, height))


#img = img.reshape(width, height, 3)
image_part = np.split(img, int(3))[0]

# plt.imshow(image_part)
# plt.show()

b, g, r, = cv.split(image_part)
i = 0
values = np.empty((width,1))
for col in b.T:

    average = np.average(col)
    values[i] = average
    print(average)
    i += 1


#plt.plot(values * height/255/3)
plt.plot(values)
plt.ylabel('średnia wartość koloru niebieskiego')
plt.xlabel('długość zdjęcia')
plt.show()
plt.imshow(image_part)
image_part_1 = cv.cvtColor(image_part, cv.COLOR_RGB2BGR)
#cv.imwrite("C:\\Users\\hiepietel\\Desktop\\b.jpg", image_part_1)

# plt.title("value")
plt.show()
plt.subplot(2, 1, 1)
plt.plot(values, '-')
plt.xlabel('średnia wartość koloru niebieskiego')
plt.ylabel('długość zdjęcia')
#plt.xticks('średnia wartość koloru niebieskiego')
#plt.yticks('długość zdjęcia')

plt.subplot(2, 1, 2)
plt.imshow(image_part)
#plt.xlabel('width')
#plt.ylabel('1/3 zdjęcia')

plt.xticks([])
plt.yticks([])
plt.show()

