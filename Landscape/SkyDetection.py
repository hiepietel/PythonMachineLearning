import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import math


hsv_l = np.array([25, 25, 127])
hsv_h = np.array([176, 224, 255])

name = 'sky'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

skies_df = pd.DataFrame()
#hists_df_center = pd.DataFrame()

width = 1920
height = 1080
for imagepath in path:

    image = cv2.imread(str(imagepath))
    image = cv2.resize(image, (width, height))
    np_image = np.copy(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #if len(image) != width and len(image[0]) != height:
    #    break
    mask = cv2.inRange(hsv, hsv_l, hsv_h)
    gray = np.where(mask == 255, 1, 0)

    line = gray.reshape((1, width*height))
    data = pd.DataFrame(line)
    firstCol = pd.DataFrame(data={'imagePath': [imagepath]})

    sky_df = pd.concat([firstCol, data], axis=1)
    skies_df = skies_df.append(sky_df, ignore_index=True, sort=False)

    print(str(imagepath))
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title("blur")
    # plt.show()
    #
    # plt.imshow(mask)
    # plt.title("blur")
    # plt.show()


skies_df.to_csv(r'sky.csv', header=True, index=False)
print("exit")


# plt.imshow(cv2.cvtColor(gray))
# plt.title("blur")
# plt.show()
#
#
#
# plt.imshow(mask)
# plt.title("blur")
# plt.show()

