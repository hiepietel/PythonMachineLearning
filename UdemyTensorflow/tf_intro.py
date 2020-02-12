import pandas as pd

df = pd.read_csv('../DATA/fake_reg.csv')
df.head()

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df)
plt.show()

from sklearn.model_selection import train_test_split

# Convert Pandas to Numpy for Keras

# Features
X = df[['feature1','feature2']].values

# Label
y = df['price'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print(X_train.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

# Final output node for prediction
model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')

model.fit(X_train,y_train,epochs=250)

loss = model.history.history['loss']

sns.lineplot(x=range(len(loss)),y=loss)
plt.title("Training Loss per Epoch");
plt.show()

print(model.metrics_names)

training_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)

test_predictions = model.predict(X_test)
pred_df = pd.DataFrame(y_test, columns=['Test Y'])

test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['Test Y', 'Model Predictions']

sns.scatterplot(x='Test Y', y='Model Predictions',data=pred_df)
plt.show()

pred_df['Error'] = pred_df['Test Y'] - pred_df['Model Predictions']
sns.distplot(pred_df['Error'], bins=50)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
print(mean_absolute_error(pred_df['Test Y'], pred_df['Model Predictions']))
print(mean_squared_error(pred_df['Test Y'], pred_df['Model Predictions']))

new_gem = [[998, 1000]]

new_gem = scaler.transform(new_gem)

print(model.predict(new_gem))

from tensorflow.keras.models import load_model

model.save('my_gem_model.h5')

later_model = load_model('my_gem_model.h5')

print(later_model.predict(new_gem))