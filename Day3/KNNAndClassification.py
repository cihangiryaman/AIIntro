"""
Cihangi Yaman 30.08.2025 Day 3
My aim for today is learning classification and KNN algorithm
This program works on Titanic dataset.
Predict a new passenger whether survive or not based on real Titanic passengers' data
"""
import warnings
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 12)

file_path = "Titanic-Dataset.csv"
data = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/titanic-dataset",
  file_path,
)
data.drop(columns=['Cabin'], inplace=True) # it has so much missing data and all we have is a single cabin.
data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Embarked'], inplace=True)
mean_fare = data['Fare'].mean()
data.fillna({'Fare':mean_fare}, inplace=True)
mean_age = data['Age'].mean()
data.fillna({'Age':mean_age}, inplace=True)
data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)


X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=91)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

y_model_test = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_model_test)
print(f"Accuracy: {accuracy}")

#example
new_passenger = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],
    'Age': [22],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25]
})
prediction = knn.predict(new_passenger)
print(f"Prediction for new passenger: {'Survived' if prediction[0] == 1 else 'Did not survive'}")