import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = keras.Sequential([
    # 케라스에게 convolutional layers를 요청. 3x3 인 64개의 필터, 28,28,1에서 1은 single byte 사용 중임을 알려줌
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # 2x2 pool, 4픽셀마다 가장 큰 것이 살아남음
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
model.summary()
