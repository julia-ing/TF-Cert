import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            print("\nloss is low so cancelling training!")
            self.model.stop_training = True

        # if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
        #     print("\nReached 99% accuracy so cancelling training!")
        #     self.model.stop_training = True


callbacks = myCallback()
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# (x_train, y_train),(x_test, y_test)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 인풋은 28x28 이미지
    keras.layers.Dense(128, activation=tf.nn.relu),  # hidden layer
    keras.layers.Dense(10, activation=tf.nn.softmax),  # 10 classes of clothing in dataset
])

# train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))

# model.compile(loss='mean_squared_error', optimizer='sgd')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,
          train_labels,
          epochs=5,
          callbacks=[callbacks])

model.evaluate(test_images, test_labels)
