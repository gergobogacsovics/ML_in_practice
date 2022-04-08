## Step 1 - Load dataset
import pandas as pd

df = pd.read_csv("../data/9/Mall_Customers.csv")

## Step 2 - X, y

## Step 3 - Split dataset into training and test splits
X_train = df

## Step 4 - Clean data & Scale the input features
from sklearn.preprocessing import StandardScaler
X_train = X_train.drop("CustomerID", axis=1)
X_train["Gender"] = (X_train["Gender"] == "Male").astype(int)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

## Step 5 - Fit model
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X_train)

from scipy.cluster.hierarchy import dendrogram

# Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plot_dendrogram(model)


model = AgglomerativeClustering(n_clusters=4).fit(X_train)
y_hat_train = model.labels_


import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
df_X_train = pd.DataFrame(scaler.inverse_transform(X_train), columns=["Gender", "Age", "Annual Income", "Spending Score"])

fig = px.scatter(df_X_train, x="Annual Income", y="Spending Score",
                 color=[str(p) for p in y_hat_train],
                 size="Age", hover_data=["Gender"])
fig.show()



## Evaluating

# Silhouette score
from sklearn.metrics import silhouette_score

score = silhouette_score(X_train, y_hat_train, metric='euclidean')

scores = []

for k in range(3, 10+1):
    model = AgglomerativeClustering(n_clusters=k).fit(X_train)
    y_hat_train = model.labels_
    scores.append(silhouette_score(X_train, y_hat_train, metric='euclidean'))

import matplotlib.pyplot as plt

plt.plot(range(3, 10+1), scores)
