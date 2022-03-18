from sklearn import datasets as ds
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler

## Step 1 - Load dataset
iris = ds.load_iris()  

## Step 2 - Split dataset into training and test splits
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

## Step 3 - Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

## Step 4 - Build the model
classifier = svm.SVC(kernel='linear')

## Step 5 - Fit the SVM model to the training data
classifier.fit(X_train, y_train)

## Step 6 - Interpret the results
print("training acc:", classifier.score(X_train, y_train))

X_test = scaler.transform(X_test)

print("test acc:", classifier.score(X_test, y_test))
