import cv2
from pathlib import Path
import matplotlib.pyplot as plt
path = Path(".")

path = path.glob("../img/*.jpg")

images=[]

for imagepath in path:

        img=cv2.imread(str(imagepath))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img, (1920, 1080))
        images.append(img)
        print(imagepath)
        hist = cv2.calcHist(img, [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0,256])
        plt.show()
print(len(images))
