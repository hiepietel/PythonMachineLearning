import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import math


blue_l = np.array([25, 25, 127])
blue_h = np.array([176, 224, 255])
#
hsv_l = np.array([0, 0, 0])
hsv_h = np.array([170, 170, 170])

name = 'sky'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

skies_df = pd.DataFrame()
#hists_df_center = pd.DataFrame()

width = 640 #1920
height = 360 #1080
sky_percent = 0.3
debug = 1
for imagepath in path:

    image = cv2.imread(str(imagepath))
    image = cv2.resize(image, (width, height))
    np_image = np.copy(image)
    image = cv2.GaussianBlur(image, (9, 9), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.erode(hsv, kernel=np.ones((3, 3)), iterations=3)
    hsv = cv2.dilate(hsv, kernel=np.ones((3, 3)), iterations=3)
    #if len(image) != width and len(image[0]) != height:
    #    break
    blue_mask = cv2.inRange(hsv, blue_l, blue_h)
    light_mask = cv2.inRange(hsv, hsv_l, hsv_h)

    blue_sky = np.where(blue_mask == 255, 0.5, 0)
    light_sky = np.where(light_mask == 255, 0, 0.5)

    gray = np.add(blue_sky, light_sky)
    sky_end = False


    for i in range(2, gray.shape[0] - 2):
        # print(np.sum(gray[i]))
        temp_sum = 0
        for j in range(i-2, i+3):
            temp_sum += np.sum(gray[j])

        # temp_sum = np.sum(gray[i - 1]) + np.sum(gray[i]) + np.sum(gray[i + 1])

        if temp_sum < sky_percent * width * 3 / 2:
            # if not sky_end:
            #     print(i)
            sky_end = True
        if sky_end:
            gray[i] = np.zeros((1, width))



    line = gray.reshape((1, width*height))
    data = pd.DataFrame(line)
    firstCol = pd.DataFrame(data={'imagePath': [imagepath]})

    sky_df = pd.concat([firstCol, data], axis=1)
    skies_df = skies_df.append(sky_df, ignore_index=True, sort=False)

    print(str(imagepath))

    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axs[0, 0].set_title('image')
    # axs[0, 1].imshow(gray, cmap='gray')
    # axs[0, 1].set_title('gray')
    # axs[1, 0].imshow(blue_sky, cmap='gray')
    # axs[1, 0].set_title('blue_sky')
    # axs[1, 1].imshow(light_sky, cmap='gray')
    # axs[1, 1].set_title('light_sky')
    #
    # plt.show()
    #
    # debug += 1
    # if debug == 10:
    #     print(debug)
    #     break

skies_df.to_csv(r'sky.csv', header=True, index=False)
print("exit")

