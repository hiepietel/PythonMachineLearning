import cv2
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

def returnColumns(end):
        col = []

        for i in range(end):
                col.append("img"+str(i))
        return col

def returnHistogram(imagepath):
    img = cv2.imread(str(imagepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (1920, 1080))

    #print(imagepath)
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])

    hist_df = pd.DataFrame(hist.reshape(-1, len(hist)), columns=returnColumns(len(hist)))

    return hist_df

name = 'desert'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

hists_df = pd.DataFrame()

for imagepath in path:
        firstCol = pd.DataFrame(data={'imagePath': [imagepath], 'isLandmark': [1]})

        hist_df = pd.concat([firstCol, returnHistogram(imagepath)], axis=1)
        hists_df = hists_df.append(hist_df,ignore_index = True, sort=False)

name = 'fake'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

for imagepath in path:
        firstCol = pd.DataFrame(data={'imagePath': [imagepath], 'isLandmark': [0]})
        #hist_df = pd.DataFrame()
        hist_df = pd.concat([firstCol, returnHistogram(imagepath)], axis=1)
        hists_df = hists_df.append(hist_df,ignore_index = True, sort=False)


#  hists_df.where(hists_df['isLandmark'] == 0).plot(kind='line', title='fake', legend=False)
#  plt.show()

hists_df.to_csv(r'hist.csv', header=True, index=False)