import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/healthcare-dataset-stroke-data-clean.csv")

independent_variables = list(set(df.columns) - {"stroke"})
target_variable = ["stroke"]

X = df[independent_variables].to_numpy()
y = df[target_variable].to_numpy()


# V1: Without scaling and without splitting
clf = LogisticRegression(random_state=0).fit(X, y)

y_hat = clf.predict(X)
print("ACC:", sum(y[:, 0] == y_hat) / len(y))


# V2: Without scaling and with splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_hat = clf.predict(X_test)
print("ACC:", sum(y_test[:, 0] == y_hat) / len(y_test))


# V3: With scaling and with splitting
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()

to_normalise = ["age", "avg_glucose_level", "bmi"]
not_to_normalise = list(set(df.columns) - set(to_normalise) - {"stroke"})

df_to_normalise = scaler.fit_transform(df[to_normalise])
df_not_to_normalise = df[not_to_normalise].to_numpy()

joined = np.concatenate((df_to_normalise, df_not_to_normalise), axis=1) 

X = joined
y = df[target_variable].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_hat = clf.predict(X_test)
print("ACC:", sum(y_test[:, 0] == y_hat) / len(y_test))
