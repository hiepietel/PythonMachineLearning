import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#img = np.zeros((200,200), np.uint8)
img = cv.imread('../img/flower.jpg', 1)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#print(img.shape)
#cv.imshow('img', img)

b, g, r, = cv.split(img)

#cv.imshow("b", b)
#cv.imshow("g", g)
#cv.imshow("r", r)

fig, ((h1, h2), (h3, h4)) = plt.subplots(2, 2)

h1.hist(img.ravel(), 256, [0,255], '.')
h2.hist(r.ravel(), 256, [0,255])
h3.hist(g.ravel(), 256, [0,255])
h4.hist(b.ravel(), 256, [0,255])
plt.xticks([])
plt.yticks([])
plt.show();

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Sharing x per column, y per row')

ax1.imshow(img)

ax2.imshow(r)
ax3.imshow(g)
ax4.imshow(b)

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
