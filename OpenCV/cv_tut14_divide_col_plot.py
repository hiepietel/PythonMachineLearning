import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import math

width = 1920
height = 1080

split = 10
splited_width = int(width/split)

folder_names = ['model_m', 'true_m', 'false_m']
mother_folder = 'img8_col2'
listt = []
imagepath ="../db/img/sky_best_not_converted/20170629_224036_HDR.jpg"
try:
    image = cv2.imread(str(imagepath))
    image = cv2.resize(image, (width, height))
    for x in range(0, width, splited_width):
        x1 = x + splited_width
        tiles = image[0:height, x:x + splited_width]
        cv2.rectangle(image, (x, 0), (x1, height), (0, 0, 0))
        listt.append(tiles)
        # tile_imagepath = '../db/img_spike/'+mother_folder+'/'+ output+'/' + str(enumerator) + '_' + str(x) + '.jpg'
        # cv2.imwrite(tile_imagepath, tiles)

except:
    print("exception")
print(str(imagepath) + " end")


plt.figure(figsize=(16,12)) # specifying the overall grid size

for i in range(10):
    plt.subplot(2, 5, i+1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(listt[i])
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
plt.show()