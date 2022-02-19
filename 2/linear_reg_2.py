import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("../data/2/Ecommerce Customers.csv")

X = df[["Avg. Session Length"]].to_numpy()
y = df["Yearly Amount Spent"].to_numpy()


plt.scatter(X[:, 0], y)
plt.show()


reg = LinearRegression().fit(X, y)


X_viz = np.array([[x] for x in range(30, 36+1)])
y_hat_viz = reg.predict(X_viz)


plt.scatter(X[:, 0], y, color="lightblue")
plt.plot(X_viz[:, 0], y_hat_viz, color="red")
plt.show()

print("The intercept is:", reg.intercept_)
print("The coefficient is:", reg.coef_[0])

print(f"Yearly Amount Spent = {reg.intercept_:.2f} + {reg.coef_[0]:.2f} * Avg. Session Length")