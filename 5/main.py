#  Original code: https://arato.inf.unideb.hu/ispany.marton/MachineLearning/iris_pca.py

from sklearn import datasets as ds
from sklearn import decomposition as decomp
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as col
from numpy import linalg as LA 
from sklearn.model_selection import train_test_split
# load dataset and partition in training and testing sets
iris = ds.load_iris()  

# Scatterplot for two input attributes
x_axis = 0  # x axis attribute (0,1,2,3)
y_axis = 1  # y axis attribute (0,1,2,3)
colors = ['blue','red','green'] # colors for target values: setosa blue, versicolor red, virginica green
n = iris.data.shape[0]
p = iris.data.shape[1]
k = iris.target_names.shape[0]
 
fig = plt.figure(1)
plt.title('Scatterplot for iris dataset')
plt.xlabel(iris.feature_names[x_axis])
plt.ylabel(iris.feature_names[y_axis])
plt.scatter(iris.data[:,x_axis],iris.data[:,y_axis],s=50,c=iris.target,cmap=col.ListedColormap(colors))
plt.show()



# Full PCA using standard numpy libraries
# computing the covariance matrix for attributes
Cov = np.cov(np.transpose(iris.data))  
# solving the eigenvalue-eigenvector problem
Cov_eig, Cov_vec = LA.eigh(Cov)    
 
# computing the principal components
PC_iris = np.dot(iris.data,Cov_vec)   
 
# checking the covariance matrix of PCs
Cov_pc = np.cov(np.transpose(PC_iris))

# dimension of the covariance matrix
dim = Cov.shape[0]  

# total variance of the covariance matrix
tot_var = np.trace(Cov)  

# checking the sum of the covariance matrix
diff = tot_var - np.sum(Cov_eig)  

# eigenvalues in decreasing order
Cov_eig = np.flipud(Cov_eig)  

# normalized variances
var_ratio = Cov_eig/tot_var    



fig = plt.figure(2) # Explained variance ratio plot
plt.title('Explained variance ratio plot')

x_pos = np.arange(dim)

plt.xticks(x_pos,x_pos+1)
plt.xlabel('Principal Components')
plt.ylabel('Variance')
plt.bar(x_pos,var_ratio, align='center', alpha=0.5)
plt.show() 


# normalized cumulative variances
cumvar_ratio = np.cumsum(Cov_eig)/tot_var    
# Explained cumulative variance ratio plot
fig = plt.figure(3)

plt.title('Explained cumulative variance ratio plot')

x_pos = np.arange(dim)
#plt.xticks(x_pos,x_pos+1)

plt.xlabel('Principal Components')
plt.ylabel('Cumulative variance')
plt.plot(x_pos,cumvar_ratio)
plt.show() 




PC_iris = np.fliplr(PC_iris) # changing the order of PCs 
fig = plt.figure(4) # Iris data in PC space

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(PC_iris[:,0],PC_iris[:,1],s=50,c=iris.target,cmap=col.ListedColormap(colors))
plt.show()



#########################################################


# Full PCA using scikit-learn
pca = decomp.PCA()
pca.fit(iris.data)

fig = plt.figure(5)

plt.title('Explained variance ratio plot')

var_ratio = pca.explained_variance_ratio_
x_pos = np.arange(len(var_ratio))

plt.xticks(x_pos,x_pos+1)
plt.xlabel('Principal Components')
plt.ylabel('Variance')
plt.bar(x_pos,var_ratio, align='center', alpha=0.5)
plt.show() 



###########################################

# PCA with limited components
pca = decomp.PCA(n_components=2)
pca.fit(iris.data)
iris_pc = pca.transform(iris.data)
class_mean = np.zeros((k,p))

for i in range(k):
    class_ind = [iris.target==i][0].astype(int)
    class_mean[i,:] = np.average(iris.data, axis=0, weights=class_ind)

PC_class_mean = pca.transform(class_mean)    
full_mean = np.reshape(pca.mean_,(1,4))
PC_mean = pca.transform(full_mean)



fig = plt.figure(6)

plt.title('Dimension reduction of the Iris data by PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(iris_pc[:,0],iris_pc[:,1],s=50,c=iris.target,
            cmap=col.ListedColormap(colors),label='Datapoints')
plt.scatter(PC_class_mean[:,0],PC_class_mean[:,1],s=50,marker='P',
            c=np.arange(k),cmap=col.ListedColormap(colors),label='Class means')
plt.scatter(PC_mean[:,0],PC_mean[:,1],s=50,c='black',marker='X',label='Overall mean')
plt.legend()
plt.show()




# PCA for train dataset and then applying for test
pr = 0.2  #  test size in probability: train:test = (1-p):p
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=pr, random_state=1)
n_train = X_train.shape[0]
n_test = X_test.shape[0]

pca = decomp.PCA(n_components=2)
pca.fit(X_train)

test_pc = pca.transform(X_test)
class_mean = np.zeros((k,p))

for i in range(k):
    class_ind = [y_test==i][0].astype(int)
    class_mean[i,:] = np.average(X_test, axis=0, weights=class_ind)
PC_class_mean = pca.transform(class_mean)    
full_mean = np.reshape(pca.mean_,(1,4))
PC_mean = pca.transform(full_mean)

fig = plt.figure(7)
plt.title('Dimension reduction of the test Iris data by PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(test_pc[:,0],test_pc[:,1],s=50,c=y_test,
            cmap=col.ListedColormap(colors),label='Datapoints')
plt.scatter(PC_class_mean[:,0],PC_class_mean[:,1],s=50,marker='P',
            c=np.arange(k),cmap=col.ListedColormap(colors),label='Class means')            
plt.scatter(PC_mean[:,0],PC_mean[:,1],s=50,c='black',marker='X',
            label='Overall mean')
plt.legend()            
plt.show()



# Comparing the complete data and partial data PCA on test dataset
X_full = np.concatenate((X_train,X_test), axis = 0)
pca.fit(X_full)
test_pc_full = pca.transform(X_test)

fig = plt.figure(8)
plt.title('Comparing  Iris data by')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(test_pc[:,0],test_pc[:,1],s=50,c=y_test,
            cmap=col.ListedColormap(colors),label='Train data PCA')
plt.scatter(test_pc_full[:,0],test_pc_full[:,1],s=50,c=y_test,
            cmap=col.ListedColormap(colors),marker='P',label='Complete data PCA')
plt.legend()            
plt.show()            

# Error between the two PCAs
rmse = np.sqrt(np.sum(np.sum((test_pc - test_pc_full)**2,axis=0))/(p*n_test))
