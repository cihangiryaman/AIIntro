"""
Cihangir Yaman 29-08-2025 Day 2
My learning goal for today is data fitting and estimating in python
This program estimate the median house value for a block with training data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('housing.csv')
x_data = df.select_dtypes(include=[np.number]).drop('median_house_value', axis=1).to_numpy()
y_data = df[['median_house_value']].to_numpy()

x_mask = np.all(np.isfinite(x_data), axis=1)
y_mask = np.isfinite(y_data).flatten()
mask = x_mask & y_mask
x_data = x_data[mask]
y_data = y_data[mask]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

model = LinearRegression()
model.fit(x_train, y_train)
y_model_train = model.predict(x_train)
y_model_test = model.predict(x_test)

print(mean_absolute_error(y_train, y_model_train))
print(mean_absolute_error(y_test, y_model_test))

plt.scatter(y_train, y_model_train, color='r', )
plt.scatter(y_test, y_model_test, color='b')
min_val = min(np.min(y_train), np.min(y_test))
max_val = max(np.max(y_train), np.max(y_test))
plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2, label='Perfect Fit (y=x)')
plt.xlabel('Actual', fontsize=20)
plt.ylabel('Model', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['train','test'], fontsize = 16)
plt.show()