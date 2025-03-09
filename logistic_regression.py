import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['species'] = iris.target

# Remove missing or impute values
print("original shape: ", df.shape)
print("number of missing values: \n", df.isnull().sum())
df = df.dropna(how = 'any')
print("shape after removing missing values: ", df.shape)

# Data statistics
print("\nData statistics: ")
print(df.describe())

# Preview data
print("\nPreview of data: ")
print(df.head())

# Feature matrix and target vector
feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
target_name = "species"
X = df.loc[:, feature_names].values
y = df.loc[:, target_name].values

print("Feature matrix X shape: ", X.shape)
print("Target vector y shape: ", y.shape)

# Split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model and fine tune hyper parameters
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

# Evaluate
print("\nEvaluation:")
print("Accuracy: ", accuracy_score(y_test, y_test_hat))
print("Confusion matrix:", confusion_matrix(y_test, y_test_hat))
print("Classification report:", classification_report(y_test, y_test_hat))
