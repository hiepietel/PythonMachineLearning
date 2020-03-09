import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../db/img/sunny_desert_s.jpg',3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()