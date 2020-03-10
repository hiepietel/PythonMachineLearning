import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#img = np.zeros((200,200), np.uint8)
#img = cv.imread('../db/img/desert/algodones_dunes_california-wallpaper-1920x1080.jpg', 1)
#img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

height = 1080
width = 1920

img = cv.imread('../db/img/sky_best_not_converted/20170629_224036_HDR.jpg', 1)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (width, height))


#img = img.reshape(width, height, 3)
half_image = np.split(img, int(3))[0]

plt.imshow(half_image)
plt.show()


hist = cv.calcHist([img],[0],None,[256],[0,256])



#hist = hist[]

# for i in range(0, 5):
#     hist[i] = 0

plt.plot(hist)
plt.title("hist")
plt.show()

print(hist)
print()
#print(img.shape)
#cv.imshow('img', img)


b, g, r, = cv.split(img)

# b_hist = cv.calcHist([b],[0],None,[256],[0,256])

b_copy = np.copy(b)


b_hist = cv.calcHist([b],[0],None,[256],[0,256])

for i in range(0, 5):
     b_hist[i] = 0

b = b/10
b = np.floor(b)

plt.imshow(b)
plt.show()


plt.plot(b_hist)
plt.title("b_hist")
plt.show()
#cv.imshow("b", b)
#cv.imshow("g", g)
#cv.imshow("r", r)

fig, ((h1, h2), (h3, h4)) = plt.subplots(2, 2)

h1.hist(img.ravel(), 256, [0, 255], '.')

h2.hist(r.ravel(), 256, [0, 255], 'r')
h3.hist(g.ravel(), 256, [0, 255])
h4.hist(b.ravel(), 256, [0, 255])

plt.xticks([])


plt.title('histograms')
plt.show()
#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# fig.suptitle('Sharing x per column, y per row')

# ax1.imshow(img)
#
# ax2.imshow(r)
# ax3.imshow(g)
# ax4.imshow(b)


plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
