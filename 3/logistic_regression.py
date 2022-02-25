import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1 - read in the datafile
df = pd.read_csv("../data/3/healthcare-dataset-stroke-data-clean.csv")

# Step 2 - separate the inputs from the outputs
independent_variables = list(set(df.columns) - {"stroke"}) # get all of the column names, except for "stroke"
target_variable = "stroke"

X = df[independent_variables]
y = df[target_variable]


# V1: Without scaling and without splitting
clf = LogisticRegression(random_state=0).fit(X, y) # Use random_state for reproducibility

y_hat = clf.predict(X)
print("Training Accuracy:", accuracy_score(y, y_hat))


# V2: Splitting the dataset into train and test but not scaling the inputs
from sklearn.model_selection import train_test_split
# Step 3 - Split the original dataset into training and test splits with a 7:3 split (70% training and 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# We can not skip Step 4 - Data cleaning, as the data has been cleaned already

# Step 5 - Fit the model to the training data
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

# Step 6 - Score the model on the test set
y_hat = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_hat))


# V3: With scaling and with splitting
from sklearn.preprocessing import StandardScaler
import numpy as np


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

scaler = StandardScaler()

# Separate the numerical values from the categorical features (0 or 1)
to_normalise = ["age", "avg_glucose_level", "bmi"]
not_to_normalise = list(set(df.columns) - set(to_normalise) - {"stroke"})

# Only transform the numerical features
df_to_normalise = scaler.fit_transform(X_train[to_normalise])
df_not_to_normalise = X_train[not_to_normalise].to_numpy()

# Merge the transformed and the untransformed columns
joined = np.concatenate((df_to_normalise, df_not_to_normalise), axis=1) 

X_train = joined

# Fit the model
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

# Measure the accuracy
# Since we transformed the training set, we also need to transform the test set

# Only transform the numerical features
df_to_normalise = scaler.fit_transform(X_test[to_normalise])
df_not_to_normalise = X_test[not_to_normalise].to_numpy()

# Merge the transformed and the untransformed columns
joined = np.concatenate((df_to_normalise, df_not_to_normalise), axis=1) 

X_test = joined

y_hat = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_hat))
