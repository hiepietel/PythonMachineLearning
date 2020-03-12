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

img = cv.imread('../db/img/sky_best_not_converted/20170701_211107~2.jpg', 1)
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


plt.plot(values*height/255/3)
plt.show()
plt.imshow(image_part)

# plt.title("value")
plt.show()
plt.subplot(2, 1, 1)
plt.plot(values, '-')
plt.title('plot and image')
plt.ylabel('average value of blue piksel')

plt.subplot(2, 1, 2)
plt.imshow(image_part)
plt.xlabel('width')
plt.ylabel('high part of image')

plt.show()

hist = cv.calcHist([img], [0], None, [256], [0, 256])
b_hist = cv.calcHist([b], [0], None, [256], [0, 256])

# plt.plot(hist)
# plt.title("hist")
#
#
# plt.plot(b_hist)
# plt.title("hist")
# plt.show()

# #hist = hist[]
#
# # for i in range(0, 5):
# #     hist[i] = 0
#
# plt.plot(hist)
# plt.title("hist")
# plt.show()
#
# print(hist)
# print()
# #print(img.shape)
# #cv.imshow('img', img)
#
#
# b, g, r, = cv.split(img)
#
# # b_hist = cv.calcHist([b],[0],None,[256],[0,256])
#
# b_copy = np.copy(b)
#
#
# b_hist = cv.calcHist([b],[0],None,[256],[0,256])
#
# for i in range(0, 5):
#      b_hist[i] = 0
#
# b = b/10
# b = np.floor(b)
#
# plt.imshow(b)
# plt.show()
#
#
# plt.plot(b_hist)
# plt.title("b_hist")
# plt.show()
# #cv.imshow("b", b)
# #cv.imshow("g", g)
# #cv.imshow("r", r)
#
# fig, ((h1, h2), (h3, h4)) = plt.subplots(2, 2)
#
# h1.hist(img.ravel(), 256, [0, 255], '.')
#
# h2.hist(r.ravel(), 256, [0, 255], 'r')
# h3.hist(g.ravel(), 256, [0, 255])
# h4.hist(b.ravel(), 256, [0, 255])
#
# plt.xticks([])
#
#
# plt.title('histograms')
# plt.show()
# #
# # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# # fig.suptitle('Sharing x per column, y per row')
#
# # ax1.imshow(img)
# #
# # ax2.imshow(r)
# # ax3.imshow(g)
# # ax4.imshow(b)
#
#
# plt.show()
#
# cv.waitKey(0)
# cv.destroyAllWindows()
