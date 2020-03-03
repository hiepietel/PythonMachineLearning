import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import math


name = 'sky_false'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

# output_name = 'sky_data'
# output_path = Path(".")
#output_path = path.glob('../' + name + '/*.jpg')

#hists_df_center = pd.DataFrame()

width = 1920
height = 1080
sky_percent = 0.3
debug = 1
size = 32
for imagepath in path:
    image = cv2.imread(str(imagepath))
    image = cv2.resize(image, (width, height))

    i = 0;
    j= 0;
    finished = False
    count_x = 0
    xount_y = 0
    while not finished:
        sky_piksels = 0
        next_i = i+size
        next_j = j+size

        #little_sky = np.array((32, 32, 3))
        little_sky = image[i:i+size, j:j+size]

        output_names = str(imagepath).split('\\')
        output_name = output_names[len(output_names) - 1]
        output_name = output_name.split('.')[0]
        output_folder = 'sky_correct_false'
        output_path = '../db/img/' + output_folder + '/' + output_name + '_' + str(i) + '_' + str(j) + '.jpg'

        cv2.imwrite(output_path, little_sky)



        if i + 2 * size < width:
            i += size
            if j + 2*size < height:
                j += size

            else:
                finished = True

    print(str(imagepath))