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
for folder in folder_names:
    output = folder[0:-2]
    path = Path(".")
    path = path.glob('../db/img_spike/'+mother_folder+'/'+folder+'/*.jpg')

    enumerator = 1
    for imagepath in path:
        try:
            image = cv2.imread(str(imagepath))
            image = cv2.resize(image, (width, height))
            for x in range(0, width, splited_width):
                x1 = x + splited_width
                tiles = image[0:height, x:x + splited_width]
                listt.append(cv2.rectangle(image, (x, 0), (x1, height), (0, 0, 0)))
                # tile_imagepath = '../db/img_spike/'+mother_folder+'/'+ output+'/' + str(enumerator) + '_' + str(x) + '.jpg'
                # cv2.imwrite(tile_imagepath, tiles)
                enumerator += 1

        except:
            print("exception")
        print(str(imagepath) + " end")

plt.figure()
for i in range(16):
    ax= plt.subplot(4, 4 , i+1)
    # im=ax.pcolor(np.random.normal(size=100).reshape([10,10]))
    plt.tight_layout()
    plt.title(i)
plt.savefig("pcolor_4x4.png")