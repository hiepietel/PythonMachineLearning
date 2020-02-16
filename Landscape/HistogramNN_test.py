import cv2
from pathlib import Path
import pandas as pd
from tensorflow.keras.models import load_model


def returnColumns(end):
    col = []

    for i in range(end):
        col.append("img" + str(i))
    return col

def returnHistogram(imagepath):
    img = cv2.imread(str(imagepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (1920, 1080))


    hist = cv2.calcHist(img, [0], None, [256], [0, 256])

    hist_df = pd.DataFrame(hist.reshape(-1, len(hist)), columns=returnColumns(len(hist)))

    return hist_df

model = load_model('../db/Model/desert_model.h5')

name = 'testDesert'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')

for imagepath in path:
    test_df = returnHistogram(imagepath).values
    #print(test)
    image = str(imagepath).split('\\')
    image = image[len(image)-1]
    message = str(image) + " "+ str(model.predict_classes(returnHistogram(imagepath).values))
    print(message)
    #print(model.predict_classes(test_df.reshape(256,1)))
    #print(image+": "+str(model.predict_classes(test_df)))

from tensorflow.keras.callbacks import TensorBoard



log_directory ='logs\\fit'

