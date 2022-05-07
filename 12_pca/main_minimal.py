#  Original code: https://arato.inf.unideb.hu/ispany.marton/MachineLearning/iris_pca.py

from sklearn import datasets as ds
from sklearn import decomposition as decomp
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as col
from numpy import linalg as LA 
from sklearn.model_selection import train_test_split

# Load dataset and partition in training and testing sets
iris = ds.load_iris()  

# Test size in probability: train:test = (1-p):p
pr = 0.2  

# Split the dataset into training and test splits
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=pr, random_state=1)

# Instantiate the PCA object with 2 components
pca = decomp.PCA(n_components=2)

# Fit the PCA object to the training set
pca.fit(X_train)

# Transform the test set with the PCA object that was fitted to the training set
test_pc = pca.transform(X_test)

# Plot the results
fig = plt.figure()
plt.title('Dimension reduction of the test Iris data by PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(test_pc[:,0],test_pc[:,1],s=50,c=y_test ,label='Datapoints')        
plt.legend()            
plt.show()
