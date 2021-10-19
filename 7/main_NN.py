from sklearn import datasets as ds
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

## Step 1 - Load dataset
iris = ds.load_iris()  

## Step 2 - Split dataset into training and test splits
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

## Step 3 - Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Step 4 - Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=[4], activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
    ])

# model.summary()
tf.keras.utils.plot_model(model, to_file="tf_model.png", show_shapes=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']) 

## Step 5 - Fit the neural network to the training data
history = model.fit(X_train, y_train, epochs=100)


## Step 6 - Plot & interpret the results
plt.plot(history.history["loss"])
plt.show()

plt.plot(history.history["accuracy"])
plt.show()

results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)


###################################################

## Step 4 - Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=[4], activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']) 

## Step 5 - Fit the neural network to the training data
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.2)

## Step 6 - Plot & interpret the results
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.show()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["acc", "val_acc"])
plt.show()

results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)


##################################################

## Step 4 - Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=[4], activation="relu"),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError()) 

## Step 5 - Fit the neural network to the training data
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.2)

## Step 6 - Plot & interpret the results
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.show()

results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)
