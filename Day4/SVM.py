"""
Cihangir Yaman 01.09.2025 Day 4
My aim for today is learning classification algorithms
This program works on Iris dataset, and uses Support Vector machines.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

encoder = LabelEncoder()

df = pd.read_csv('Iris.csv')
df = df.drop(columns='Id')
df['Species'] = encoder.fit_transform(df['Species'])

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target = 'Species'

x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)
