from sklearn import datasets as ds
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

## Step 1 - Load dataset
ds = ds.load_diabetes()  

## Step 2 - Split dataset into training and test splits
X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.3, random_state=1)

## Step 3 - Scale outputs
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train.reshape(-1,1))[:, 0]
y_test = scaler.transform(y_test.reshape(-1,1))[:, 0]


## Step 4 - Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=[10], activation="relu"),
    tf.keras.layers.Dense(3)
    ])

# model.summary()
tf.keras.utils.plot_model(model, to_file="tf_model.png", show_shapes=True)

model.compile(optimizer='adam',
              loss="mean_absolute_error",
              metrics=["mae", "mse"])

## Step 5 - Fit the neural network to the training data
history = model.fit(X_train, y_train, epochs=100)


## Step 6 - Plot & interpret the results
plt.plot(history.history["loss"])
plt.show()

plt.plot(history.history["mae"])
plt.show()

plt.plot(history.history["mse"])
plt.show()

results = model.evaluate(X_test, y_test)
print("results:", results)

