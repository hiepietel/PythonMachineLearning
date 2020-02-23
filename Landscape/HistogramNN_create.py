import cv2
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

width = 1920
height = 1080
offset = 0.35


def returnColumns(end):
        col = []

        for i in range(end):
                col.append("img"+str(i))
        return col

def returnHistogramCenter(imagepath):
    img = cv2.imread(str(imagepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))

    min = int(offset * height)
    max = int((1-offset) * height)
    ran = range(min, max)
    center_img = img[ran, :]

    plt.imshow(center_img)
    path = str(imagepath)
    path = path[0:-4]
    plt.savefig(path+"_center.png", figsize=(width/100, height/100))
    #plt.savefig('foo.png')
    plt.close()

    #print(imagepath)
    center_hist = cv2.calcHist(center_img, [0], None, [256], [0, 256])

    center_hist_df = pd.DataFrame(center_hist.reshape(-1, len(center_hist)), columns=returnColumns(len(center_hist)))

    return center_hist_df


def returnHistogram(imagepath):
    img = cv2.imread(str(imagepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (width, height))

    min = int(0.4 * height)
    max = int(0.6 * height)
    ran = range(min, max)
    centerImg = img[ran, :]

    #print(imagepath)
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])

    hist_df = pd.DataFrame(hist.reshape(-1, len(hist)), columns=returnColumns(len(hist)))

    return hist_df

name = 'desert'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

hists_df = pd.DataFrame()
hists_df_center = pd.DataFrame()
for imagepath in path:
        firstCol = pd.DataFrame(data={'imagePath': [imagepath], 'isLandmark': [1]})

        hist_df = pd.concat([firstCol, returnHistogram(imagepath)], axis=1)
        hists_df = hists_df.append(hist_df, ignore_index = True, sort=False)

        hist_df_center = pd.concat([firstCol, returnHistogramCenter(imagepath)], axis=1)
        hists_df_center = hists_df_center.append(hist_df_center, ignore_index = True, sort=False)


name = 'fake'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

for imagepath in path:
        firstCol = pd.DataFrame(data={'imagePath': [imagepath], 'isLandmark': [0]})
        #hist_df = pd.DataFrame()
        hist_df = pd.concat([firstCol, returnHistogram(imagepath)], axis=1)
        hists_df = hists_df.append(hist_df,ignore_index = True, sort=False)

        hist_df_center = pd.concat([firstCol, returnHistogramCenter(imagepath)], axis=1)
        hists_df_center = hists_df_center.append(hist_df_center, ignore_index = True, sort=False)
#  hists_df.where(hists_df['isLandmark'] == 0).plot(kind='line', title='fake', legend=False)
#  plt.show()

hists_df.to_csv(r'hist.csv', header=True, index=False)
hists_df_center.to_csv(r'hist_center.csv', header=True, index=False)