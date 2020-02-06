from matplotlib import pyplot as plt
import cv2

img = cv2.imread('43.png' -1)
cv2.imshow('image', img)
img = cv2.cvtcolor(img, cv2.COLOR_BGR2RGB) #converting
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()




cv2.waitKey(0)
cv2.destroyAllWindows()

##mamny image
img2 = img
img1 = img1
titles = ["first","second", "third"]
images = [img, img1, img2]
for i in range(6):
	plt.subplot(2,3,plt.imshow(images[i], 'gray') #rows, cols, index 
	plt.title(titles[i])
	
plt.show()