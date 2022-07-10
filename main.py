# importing tensorflow and tensorflow_datasets
import tensorflow as tf
import tensorflow_datasets as tfds

# importing additional libraries
import numpy as np
import matplotlib.pyplot as plt
import logging as log
import math as m

# setting up the logging
logger = tf.get_logger()
logger.setLevel(log.ERROR)

# download the dataset mnist from tensorflow_datasets with info
data, info = tfds.load('mnist', as_supervised=True, with_info=True)
# split the dataset into train and test
train_data, test_data = data['train'], data['test']
# define the class names
class_names = [
    'zero', 'one', 'two', 'three',
    'four', 'five', 'six','seven',
    'eight', 'nine'
]
# get the number of train and test samples
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

# define the function to preprocess the data
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label
# use the normalize function to preprocess the data
train_data = train_data.map(normalize)
test_data = test_data.map(normalize)

# repeat and shuffle only the training data
# then batch both the training and test data
BATCH_SIZE = 32
train_data = train_data.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

# define the neural network model 
# using the tf.keras.Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
model.fit(train_data, epochs=10, steps_per_epoch=int(m.ceil(num_train_examples/BATCH_SIZE)))
# evaluate the model getting the test loss and accuracy
test_loss, test_acc = model.evaluate(test_data, steps=int(m.ceil(num_test_examples/BATCH_SIZE)))