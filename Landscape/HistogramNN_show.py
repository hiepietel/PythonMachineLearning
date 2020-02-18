import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
path = 'hist.csv'
rand_amount = 5
hists_df = pd.read_csv(path)

landscape_df =  hists_df.where(hists_df['isLandmark'] == 1).dropna().drop('imagePath', axis=1).drop('isLandmark', axis=1)
for i in range(len(landscape_df.index)):
    landscape_df.xs(i).plot(title='histogram of deserts',kind='line', xticks=[0, 127, 255])
plt.ylim((0,500))
plt.show()

land_rand = np.random.randint(rand_amount, high=len(landscape_df.index), size=(rand_amount,))
for i in range(len(land_rand)):
    landscape_df.xs(land_rand[i]).plot(title='histogram of 5 random deserts', kind='line', xticks=[0, 127, 255])
plt.ylim((0,500))
plt.show()


fake_df = hists_df.where(hists_df['isLandmark'] == 0).dropna().drop('imagePath', axis=1).drop('isLandmark', axis=1)
for i in range(len(fake_df.index)):
    fake_df.xs(i+len(landscape_df.index)).plot(title='histogram of fakes',kind='line', xticks=[0, 127, 255])
plt.ylim((0,500))
plt.show()

fake_rand = np.random.randint(rand_amount, high=len(fake_df.index), size=(rand_amount,))
for i in range(len(fake_rand)):
    fake_df.xs(fake_rand[i]+len(landscape_df.index)).plot(title='histogram of 5 random fakes', kind='line', xticks=[0, 127, 255])
plt.ylim((0,500))
plt.show()