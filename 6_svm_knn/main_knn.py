from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import pandas as pd

## Step 1 - Load dataset
df = pd.read_csv("../data/5/data.csv")

## Step 2 - Split dataset into X and y
X = df.iloc[:, 0:56]
y = df.iloc[:, 57]

## Step 3 - Split dataset into training and test splits
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
                                                    shuffle=True, random_state=0)

## Step 4 - Build the model
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

print("Train ACC: ", classifier.score(X_train, y_train))
print("Test ACC: ", classifier.score(X_test, y_test))


## Grid Search
### Specify the hyperparameters to check
classifier = KNeighborsClassifier()
leaf_size = [i for i in range(1, 10+1)]
n_neighbors = [i for i in range(1, 10+1)]
p = [1,2]

hyperparameters = {
    "leaf_size": leaf_size,
    "n_neighbors": n_neighbors,
    "p": p
}

### Run Grid Search
search = GridSearchCV(classifier, hyperparameters, cv=2)
best_model = search.fit(X_train, y_train)

### Get the best hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

## Fit a KNN with the best hyperparameters found during Grid Search
classifier = KNeighborsClassifier(n_neighbors=3, leaf_size=1, p=1)
classifier.fit(X_train, y_train)
print("Train ACC: ", classifier.score(X_train, y_train))
print("Test ACC: ", classifier.score(X_test, y_test))
