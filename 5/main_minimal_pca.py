#  Original code: https://arato.inf.unideb.hu/ispany.marton/MachineLearning/iris_pca.py

from sklearn import datasets as ds
from sklearn import decomposition as decomp
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

# load dataset and partition in training and testing sets
iris = ds.load_iris()  


# PCA for train dataset and then applying for test
pr = 0.2  #  test size in probability: train:test = (1-p):p
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=pr, random_state=1)


pca = decomp.PCA(n_components=2)
pca.fit(X_train)


test_pc = pca.transform(X_test)


plt.title('Dimension reduction of the test Iris data by PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(test_pc[:,0],test_pc[:,1],s=50,c=y_test,label='Datapoints')                
plt.show()
