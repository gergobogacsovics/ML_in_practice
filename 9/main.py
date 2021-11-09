## Step 1 - Load dataset
import pandas as pd

df = pd.read_csv("data/Mall_Customers.csv")

## Step 2 - X, y

## Step 3 - Split dataset into training and test splits
import numpy as np

train_percentage = 0.7
train_size = int(df.shape[0] * train_percentage)

X_train_indices = np.random.choice(df.shape[0], train_size, replace=False)
assert len(np.unique(X_train_indices)) == train_size

X_test_indices = list(set(range(df.shape[0])) - set(X_train_indices))

assert len(X_train_indices) + len(X_test_indices) == df.shape[0]
assert len(set(range(df.shape[0])) - set(list(X_train_indices) + X_test_indices)) == 0

X_train = df.iloc[X_train_indices]
X_test = df.iloc[X_test_indices]

## Step 4 - Clean data & Scale the input features
from sklearn.preprocessing import StandardScaler
X_train = X_train.drop("CustomerID", axis=1)
X_train["Gender"] = (X_train["Gender"] == "Male").astype(int)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

## Step 5 - Fit model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distances = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_train)
    distances.append(kmeans.inertia_)

plt.plot(range(2, 20), distances, "*-")
plt.grid(True)
plt.title("Elbow method")
plt.xticks(range(2, 20))
plt.show()

kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train)
y_hat_train = kmeans.predict(X_train)

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
df_X_train = pd.DataFrame(scaler.inverse_transform(X_train), columns=["Gender", "Age", "Annual Income", "Spending Score"])

fig = px.scatter(df_X_train, x="Annual Income", y="Spending Score",
                 color=[str(p) for p in y_hat_train],
                 size="Age", hover_data=["Gender"])
fig.show()



## Step 6 - Interpret the results
X_test = X_test.drop("CustomerID", axis=1)
X_test["Gender"] = (X_test["Gender"] == "Male").astype(int)
X_test = scaler.transform(X_test)

y_hat_test = kmeans.predict(X_test)

df_X_test = pd.DataFrame(scaler.inverse_transform(X_test), columns=["Gender", "Age", "Annual Income", "Spending Score"])

fig = px.scatter(df_X_test, x="Annual Income", y="Spending Score",
                 color=[str(p) for p in y_hat_test],
                 size="Age", hover_data=["Gender"])
fig.show()

