from sklearn.datasets import load_breast_cancer

## Step 1 - Load dataset
dataset = load_breast_cancer()
feature_names = dataset.feature_names

## Step 2 - X,y
X, y = dataset.data, dataset.target

## Step 3 - Split dataset into training and test splits
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)


## Step 4 - Scale the input features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

## Step 5 - Fit the model to the training data
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.svm import SVC

models = [
    {
     "name": "logistic regression",
     "model": LogisticRegression(),
     "parameters": {"penalty": ["none", "l1", "l2"], "C": [0.1, 1.0, 1.5, 2.0, 5.0]}
    },
        {
     "name": "SVM",
     "model": SVC(),
     "parameters": {"kernel": ["linear", "poly", "rbf"], "C": [0.1, 1.0, 1.5, 2.0, 5.0]}
    }
]

results = []

for m in models:
    print("Running search for ", m["name"])
    
    # Do a GridSearchCV for the given model
    search = GridSearchCV(m["model"], m["parameters"], cv=5)
    search.fit(X_train, y_train)
    
    print("ACC mean:", search.cv_results_["mean_test_score"])
    print("ACC std:", search.cv_results_["std_test_score"])
    print("ACC max:", np.nanmax(search.cv_results_["mean_test_score"]))
    
    # Store the results
    df = pd.DataFrame.from_dict(search.cv_results_)
    results.append({"model": m["name"], "results": df,
                    "acc max": np.nanmax(search.cv_results_["mean_test_score"]),
                    "best_params": search.best_params_})


classifier = LogisticRegression(C=1.0, penalty="l2")
classifier.fit(X_train, y_train)
print("Training acc:", classifier.score(X_train, y_train))

## Step 6 - Interpret the results
X_test = scaler.transform(X_test)

print("Test acc:", classifier.score(X_test, y_test))

from sklearn.metrics import precision_recall_fscore_support
print(precision_recall_fscore_support(y_test, classifier.predict(X_test),
                                      pos_label=1, average="binary"))


from sklearn import metrics
metrics.plot_roc_curve(classifier, X_test, y_test) 
