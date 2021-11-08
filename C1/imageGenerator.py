import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from google.colab import files
from keras.preprocessing import image

model = tf.keras.Sequential([
    # 색상이 있기 때문에 rgb -> 3 byte
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


train_generator = ImageDataGenerator(rescale=1./255)
test_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_generator.flow_from_directory(
    train_dir,  # 1,024 images in the training directory
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'  # 두 가지 옵션
)

validation_generator = test_generator.flow_from_directory(
    validation_dir,  # 256 images in the validation directory
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'  # 두 가지 옵션
)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,  # 128x8 = 1024
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8,  # 32x8 = 256
    verbose=2
)

uploaded = files.upload()

for fn in uploaded.keys():
    # predicting images
    path = '/content/' + fn
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + " is a human")
    else:
        print(fn + " is a horse")
