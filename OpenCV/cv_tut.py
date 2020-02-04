import cv2

#print(cv2.__version__)
img = cv2.imread('image.jpg', 0)
print(img)

cv2.imshow('image',img)

k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('image_copy.png', img)