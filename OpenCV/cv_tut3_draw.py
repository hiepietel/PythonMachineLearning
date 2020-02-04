import numpy as np
import cv2

#img = cv2.imread('image.jpg', 1)
img = np.zeros([512, 512, 3], np.uint8) # black image

img = cv2.line(img, (0,0), (255,255), (255, 0, 0), 5) # image
img = cv2.arrowedLine(img, (0,255), (255,255), (255, 0, 0), 5) # arrow

img = cv2.rectangle(img, (10,10), (510,210), (255,255,0), 10) #rect 

img = cv2.circle(img, (200,200), 50, (0, 255,0 ), -1) # circle

font = cv2.FONT_HERSHEY_SIMPLEX #font 
img = cv2.putText(img, 'ELO', (300, 200), font, 4,(255,255,255), 10, cv2.LINE_4) #text
cv2.imshow('image', img) 

cv2.waitKey(0)
cv2.destroyAllWindows()