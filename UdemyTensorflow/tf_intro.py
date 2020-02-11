import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../DATA/fake_reg.csv')

print(df)

sns.pairplot(df)
plt.show()
from sklearn.model_selection import train_test_split


X = df[['feature1', 'feature2']].values

y = df['price'].values
print('X')
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#scaled to 0.0 to 1.0

#create model with keras syntax

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#model = Sequential([Dense(4, activation='relu'),
#                    Dense(2, activation='relu'),
#                    Dense(1)])

#as above
model = Sequential()

model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(x = X_train, y=y_train, epochs= 250)

loss = pd.DataFrame(model.history.history)

plt.plot(loss)
plt.show()

print(model.evaluate(X_test, y_test, verbose=0))
print(model.evaluate(X_train, y_train, verbose=0))

test_predictions = model.predict(X_test)
print(test_predictions)

test_predictions = pd.Series(test_predictions.reshape((300,)))
print(test_predictions)
pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['Test True Y', 'Model Predictions']
print(pred_df)

sns.scatterplot(x='Test True Y', y='Model Predictions', data=pred_df)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions']))
print(mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions']))