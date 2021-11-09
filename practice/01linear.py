import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])


def test_model(y_new):
    xs = [1, 2, 3, 4, 5, 6]
    ys = [1, 1.5, 2, 2.5, 3, 3.5]
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=100)
    return model.predict(y_new)[0]


prediction = test_model([7.0])
print(prediction)

# model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
# model.compile(optimizer='sgd', loss='mean_squared_error')
#
# xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
#
# model.fit(xs, ys, epochs=500)
#
# print(model.predict([10.0]))
