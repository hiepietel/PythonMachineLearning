#https://www.youtube.com/watch?v=eLTLtUVuuy4
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('../db/img/cyberpunk_s.jpg')
lane_image = np.copy(image)

gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)


plt.imshow(lane_image)
plt.show()
cv2.imshow("result", gray)
cv2.waitKey(0)
