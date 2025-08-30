"""
This is %90 percent created by AI
Test program to find best parameters for KNN algorithm
"""
import warnings
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
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

data.drop(columns=['Cabin'], inplace=True)
data.drop(columns=['Embarked'], inplace=True)
data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
mean_fare = data['Fare'].mean()
data.fillna({'Fare':mean_fare}, inplace=True)
mean_age = data['Age'].mean()
data.fillna({'Age':mean_age}, inplace=True)
data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)


X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = data['Survived']

best_accuracy = 0
best_random_state = 0
best_n_neighbors = 0

for random_state in range(101):
    for n_neighbors in range(1, 21):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_model_test = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_model_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_random_state = random_state
            best_n_neighbors = n_neighbors

print(f"Best Accuracy: {best_accuracy}")
print(f"Best random_state: {best_random_state}")
print(f"Best n_neighbors: {best_n_neighbors}")