from sklearn import datasets as ds
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Step 1 - Load dataset
iris = ds.load_iris()  

# Step 2 - Split dataset into training and test splits
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

# Step 3 - Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4 - Build the model
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(random_state=0, max_iter=100, hidden_layer_sizes=(32, 16), batch_size=32, solver="adam", activation="relu")

# Step 5 - Fit the neural network to the training data
clf.fit(X_train, y_train)

# Step 6 - Interpret the results
from sklearn.metrics import accuracy_score, precision_score

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print("Training ACC:", accuracy_score(y_true=y_train, y_pred=y_pred_train))
print("Test ACC:", accuracy_score(y_true=y_test, y_pred=y_pred_test))
