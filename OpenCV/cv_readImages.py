import cv2
from pathlib import Path

path = Path(".")

path = path.glob("../img/*.jpg")

images=[]

for imagepath in path:

        img=cv2.imread(str(imagepath))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(200,200))
        images.append(img)
        print(imagepath)
print(len(images))