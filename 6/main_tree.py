#  Original code: https://arato.inf.unideb.hu/ispany.marton/MachineLearning/2020%20fall/tree_of_spambase.py

import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

import pandas as pd

df = pd.read_csv("data.csv")

X = df.iloc[:, 0:56]
y = df.iloc[:, 57]

input_names = X.columns
target_names = ['not spam','spam']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
                                                    shuffle=True, random_state=0)


## GINI
crit = 'gini'
depth = 4

classifier = DecisionTreeClassifier(criterion=crit,max_depth=depth)

classifier.fit(X_train, y_train)
score_train = classifier.score(X_train, y_train)
score_test = classifier.score(X_test, y_test)

y_pred_gini = classifier.predict(X_test)

# Visualization
fig = plt.figure(1, figsize=(16,10), dpi=100)
plot_tree(classifier, feature_names=input_names, 
               class_names=target_names,
               filled=True, fontsize=6)


#fig.savefig('spambase_tree_gini.png')



## Entropy
crit = 'entropy'
depth = 4


class_tree = DecisionTreeClassifier(criterion=crit, max_depth=depth)

class_tree.fit(X_train, y_train)
score_entropy = class_tree.score(X_train, y_train)
score_test = class_tree.score(X_test, y_test)

y_pred_entropy = class_tree.predict(X_test);

# Visualizing
fig = plt.figure(2,figsize = (16,10),dpi=100)
plot_tree(class_tree, feature_names=input_names, 
               class_names=target_names,
               filled=True, fontsize=6)


#fig.savefig('spambase_tree_entropy.png'); 
