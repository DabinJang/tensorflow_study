import tensorflow as tf
import tensorflow.keras.layers as tfl
# import numpy as np
# import matplotlib.pyplot as plt

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data downscaling
x_train, x_test = x_train/255.0, x_test/255.0

# Make model
model = tf.keras.models.Sequential([
    tfl.Flatten(input_shape=(28, 28)),
    tfl.Dense(units=128, activation='relu'),
    tfl.Dropout(0.2),
    tfl.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)
