from sklearn import datasets as ds
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def build_model(hidden_layers=[32], optimizer="adam", metrics=["accuracy"], dropout_p=0):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(hidden_layers[0], input_shape=[4],
                                        activation="relu"))
    
    for neurons in hidden_layers[1:]:
        model.add(tf.keras.layers.Dense(neurons, input_shape=[4],
                                        activation="relu"))
        if dropout_p > 0:
            model.add(tf.keras.layers.Dropout(dropout_p))
    
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=metrics) 
    
    return model

## Step 1 - Load dataset
iris = ds.load_iris()  

## Step 2 - Split dataset into training and test splits
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

## Step 3 - Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Step 4 - Build the model
model = build_model(hidden_layers=[32, 32], dropout_p=0.5)

## Step 5 - Fit the neural network to the training data
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2)

## Step 6 - Plot & interpret the results
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "validation loss"])
plt.show()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["acc", "validation acc"])
plt.show()

results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)

