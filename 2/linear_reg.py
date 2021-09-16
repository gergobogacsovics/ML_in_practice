# Original code: https://arato.inf.unideb.hu/ispany.marton/MachineLearning/2021%20fall/basic_linear_regression.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Default parameters
n = 1000  # sample size
b = 3   # intercept
w_1 = 2   # slope
sigma = 1 # error 



#  Generating random sample  
x = np.random.normal(0, 1, n)   #  standard normally distributed input
eps = np.random.normal(0, sigma, n)  #  random error
y = b + w_1*x + eps   #  regression equation




# Scatterplot with regression line 
plt.title('Scatterplot of data with regression line')
plt.xlabel('x input')
plt.ylabel('y output')
#xmin = min(x)-0.3
#xmax = max(x)+0.3
#ymin = b + w * xmin
#ymax = b + w * xmax
plt.scatter(x, y, color="blue")  #  scatterplot of data
#plt.plot([xmin,xmax],[ymin,ymax],color='red')  #  plot of regression line
plt.show() 




# Fitting linear regression
reg = LinearRegression()  # instance of the LinearRegression class
X = np.expand_dims(x, 1) #x.reshape(1, -1).T; # reshaping 1D array to 2D one
reg.fit(X, y)   #  fitting the model to data
b_hat = reg.intercept_  #  estimated intercept
w_1_hat = reg.coef_[0]   #  estimated slope


# Evaluating
R2 = reg.score(X, y)   #  R-square for model fitting
y_pred = reg.predict(X)  #  prediction of the target

mse=mean_squared_error(y, y_pred)


# V2

# Computing the regression coefficients by using basic numpy
# Compare estimates below with b0hat and b1hat
reg_coef = np.ma.polyfit(x, y, 1)  



# Printing the results
print(f'Estimated slope:{w_1_hat:6.4f} (True slope:{w_1})')
print(f'Estimated intercept:{b_hat:6.4f} (True intercept:{b})')
print(f'R-square for goodness of fit:{R2:6.4f}')




# Scatterplot for data with true and estimated regression line
plt.title('Scatterplot of data with regression lines')
plt.xlabel('x input')
plt.ylabel('y output')
xmin = min(x)-0.3
xmax = max(x)+0.3
ymin = b + w_1*xmin
ymax = b + w_1*xmax
plt.scatter(x, y,color="blue")
plt.plot([xmin,xmax],[ymin,ymax], color='black')
ymin = b_hat + w_1_hat*xmin
ymax = b_hat + w_1_hat*xmax
plt.plot([xmin,xmax],[ymin,ymax],color='red')
plt.show()




# Scatterplot for target prediction
plt.title('Scatterplot for prediction')
plt.xlabel('True target')
plt.ylabel('Predicted target')
ymin = min(y)-1
ymax = max(y)+1
plt.scatter(y,y_pred,color="blue")
plt.plot([ymin,ymax],[ymin,ymax],color='red')
plt.show()

