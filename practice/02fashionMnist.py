import tensorflow as tf
from tensorflow import keras
import requests
requests.packages.urllib3.disable_warnings()
import ssl

"""
mnist / DNN
"""

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            print("\nloss is low so cancelling training!")
            self.model.stop_training = True

        # if logs.get('acc') is not None and logs.get('acc') > 0.99:
        #     print("\nReached 99% accuracy so cancelling training!")
        #     self.model.stop_training = True


callbacks = myCallback()
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# (x_train, y_train),(x_test, y_test)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.fit(train_images,
          train_labels,
          epochs=5,
          callbacks=[callbacks])

model.evaluate(test_images, test_labels)
