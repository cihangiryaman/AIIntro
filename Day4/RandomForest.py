import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
model = RandomForestClassifier(n_estimators=100, random_state=42)

df = pd.read_csv('Iris.csv')
df = df.drop(columns='Id')
df['Species'] = encoder.fit_transform(df['Species'])

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target = 'Species'

x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print(f"Model Accuracy: {score}")

importances = model.feature_importances_

i = 0
for feature in features:
    print(feature, ':', importances[i])
    i = i + 1