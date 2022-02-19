import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/2/Ecommerce Customers.csv")

X = df[["Avg. Session Length"]].to_numpy()
y = df["Yearly Amount Spent"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

del X, y

plt.scatter(X_train[:, 0], y_train)
plt.show()


reg = LinearRegression().fit(X_train, y_train)


X_viz = np.array([[x] for x in range(30, 36+1)])
y_hat_viz = reg.predict(X_viz)


plt.title("Training set")
plt.scatter(X_train[:, 0], y_train, color="lightblue")
plt.scatter(X_test[:, 0], y_test, color="navy")
plt.plot(X_viz[:, 0], y_hat_viz, color="red")
plt.show()

print("The intercept is:", reg.intercept_)
print("The coefficient is:", reg.coef_[0])

print(f"Yearly Amount Spent = {reg.intercept_:.2f} + {reg.coef_[0]:.2f} * Avg. Session Length")