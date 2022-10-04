# tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # split into testing and training data
# train_images.shape
# train_images[0,23,23]
# train_labels[:10]

class_names = ['T-shirt', 'Pants', 'Jacket', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Shoe', 'Bag', 'Boot']

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'), # hidden layer (2)
    keras.layers.Dense(256, activation='relu'), # hidden layer (3)
    keras.layers.Dense(80, activation='relu'),  # hidden layer (4)
    keras.layers.Dense(10, activation='softmax')# output layer (5)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10) # we pass the data, labels, and epochs and watch the magic happen!

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy: ' + str(test_acc))