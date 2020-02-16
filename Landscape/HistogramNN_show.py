import pandas as pd
import matplotlib.pyplot as plt
path = 'fakehist.csv'

hists_df = pd.read_csv(path)

landscape_df =  hists_df.where(hists_df['isLandmark'] == 1).dropna().drop('imagePath', axis=1).drop('isLandmark', axis=1)
for i in range(len(landscape_df.index)):
    landscape_df.xs(i).plot(title='histogram of deserts',kind='line', xticks=[0, 127, 255])
plt.ylim((0,500))
plt.show()

fake_df = hists_df.where(hists_df['isLandmark'] == 0).dropna().drop('imagePath', axis=1).drop('isLandmark', axis=1)
for i in range(len(fake_df.index)):
    fake_df.xs(i+len(landscape_df.index)).plot(title='histogram of fakes',kind='line', xticks=[0, 127, 255])
plt.ylim((0,500))
plt.show()