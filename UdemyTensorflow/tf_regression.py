import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../DATA/kc_house_data.csv')

print(df.isnull().sum())
print(df.describe().transpose())

plt.figure(figsize=(10,6))
sns.distplot((df['price']))
plt.plot()