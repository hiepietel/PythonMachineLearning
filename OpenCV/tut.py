#pod img np zeros
#cv.rectangle(img, (0, 100), (200, 200), (255), -1)
#cv.rectangle(img, (50,50), (150, 150), (234), -1)

img = cv.imread("lena.jpg",0)
######################
img = cv.imread("lena.jpg")
b, g, r, = cv.split(img)

cv.imshow("b", b)
cv.imshow("g", g)
cv.imshow("r", r)

plt.hist(b.ravel(), 256, [0,255])
plt.hist(g.ravel(), 256, [0,255])
plt.hist(r.ravel(), 256, [0,255])

plt.hist(img.ravel(), 256, [0,255])
plt.show()
#na koniec 
img 
hist = cv.calcHist([img], [0],None, [256], [0,256])
plt.plot(hist)