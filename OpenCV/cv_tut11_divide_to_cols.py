import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import math

width = 1920
height = 1080

split = 48
splited_width = int(width/split)

name = 'model_complete'
path = Path(".")
path = path.glob('../db/img_col/'+name+'/*.jpg')

enumerator = 1
for imagepath in path:
    try:
        image = cv2.imread(str(imagepath))
        image = cv2.resize(image, (width, height))
        for x in range(0, width, splited_width):
            x1 = x + splited_width
            tiles = image[0:height, x:x + splited_width]
            cv2.rectangle(image, (x, 0), (x1, height), (0, 0, 0))
            tile_imagepath = '../db/img_col/splited/' + str(enumerator) + '_' + str(x) + '.jpg'
            cv2.imwrite(tile_imagepath, tiles)
            enumerator += 1

    except:
        print("exception")
    print(str(imagepath) + " end")
# cv2.waitKey(0)
# cv2.destroyAllWindows()