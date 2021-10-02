import pandas as pd

df = pd.read_csv("data/train.csv")

print(df.head())
print(df.describe())
print(df.info())


from pandas_profiling import ProfileReport

profile = ProfileReport(df,
                        title="Report",
                        explorative=True)

profile.to_file("myreport.html")


X = df.loc[:, df.columns != "Survived"]
y = df["Survived"]

def get_dummies(df, column):
    dummies = pd.get_dummies(df[column],
                             drop_first=True,
                             prefix=column) # !
    
    return pd.concat([df, dummies], axis=1).drop(
        column, axis=1)

def clean(df, mean_age):
    df = df.drop("PassengerId", axis=1)
    df = df.drop("Name", axis=1)
    df = df.drop("Ticket", axis=1)
    df = df.drop("Cabin", axis=1)
    
    df.loc[df["Age"].isna(), "Age"] = mean_age
    
    
    df = get_dummies(df, "Pclass")
    df = get_dummies(df, "Sex")
    df = get_dummies(df, "Embarked")
    
    return df

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

MEAN_AGE = int(X_train["Age"].mean())

X_train = clean(X_train, mean_age=MEAN_AGE)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X_train.loc[:, ["Age", "Fare"]] = scaler.fit_transform(
    X_train.loc[:, ["Age", "Fare"]])

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)

model.fit(X_train, y_train)

X_test = clean(X_test, mean_age=MEAN_AGE)
X_test.loc[:, ["Age", "Fare"]] = scaler.transform(
    X_test.loc[:, ["Age", "Fare"]])

y_hat = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_hat)