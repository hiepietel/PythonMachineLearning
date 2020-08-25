import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#img = np.zeros((200,200), np.uint8)
img = cv.imread('../db/img/sky_best_not_converted/20170629_224036_HDR.jpg', 1)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)



imgg = cv.imread('../db/img/sky_best_not_converted/20170629_224036_HDR.jpg', 1)
imgg = cv.cvtColor(imgg, cv.COLOR_BGR2RGB)
hist = cv.calcHist([imgg],[0],None,[256],[0,256])

#hist = hist[]


print(hist)
print()
#print(img.shape)
#cv.imshow('img', img)

b, g, r, = cv.split(img)

#cv.imshow("b", b)
#cv.imshow("g", g)
#cv.imshow("r", r)

# fig, (h1, h2) = plt.subplots(2, 1)
#
# h2.hist(img.ravel(), 256, [0, 255], '.')
#
# h1.imshow(img)
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
#
# plt.show()

# plt.title("value")
plt.show()
plt.subplot(2, 1, 1)
plt.title('przykładowe zdjęcie i jego histogram')
plt.hist(img.ravel(), 256, [0, 255], '-')
plt.xticks([])
plt.yticks([])
plt.ylabel('histogram')
#plt.title('plot and image')
#plt.ylabel('average value of blue piksel')

plt.subplot(2, 1, 2)
plt.imshow(imgg, aspect="auto")
plt.xticks([])
plt.yticks([])
plt.ylabel('zdjęcie')
#plt.xlabel('width')
#plt.ylabel('high part of image')
# fig.suptitle('Sharing x per column, y per row')
plt.show()


# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# fig.suptitle('Sharing x per column, y per row')
#
# ax1.imshow(img)
#
# ax2.imshow(r)
# ax3.imshow(g)
# ax4.imshow(b)
#
# plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
