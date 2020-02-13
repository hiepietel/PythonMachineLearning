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

print(sns.countplot(df['bedrooms']))
print(df.corr()['price'].sort_values())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='sqft_living', data=df)
plt.show()

sns.boxplot(x='bedrooms', y='price', data=df)
plt.show()

print(df.columns)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='long', y='lat', data=df, hue='price')
plt.show()
print(df.sort_values('price', ascending=False).head(20))
print(len(df)*0.01)

non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]
plt.figure(figsize=(12, 8))
sns.scatterplot(x='long', y='lat', data=non_top_1_perc, edgecolor=None, alpha=0.2, palette='RdYlGn', hue='price')
plt.show()

sns.boxplot(x='waterfront', y='price', data=df)
