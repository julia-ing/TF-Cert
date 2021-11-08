import os
import zipfile
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import shuffle
from shutil import copyfile
from os import getcwd

path_cats_and_dogs = "C:/TensorFlow Coursera/cats-and-dogs.zip"

local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('E:/TensorFlow Coursera')
zip_ref.close()

print(len(os.listdir('C:/TensorFlow Coursera/PetImages/Cat/')))
print(len(os.listdir('C:/TensorFlow Coursera/PetImages/Dog/')))

# Expected Output:
# 1500
# 1500

# Use os.mkdir to create your directories
# You will need a directory for cats-v-dogs, and subdirectories for training
# and testing. These in turn will need subdirectories for 'cats' and 'dogs'
try:
    os.mkdir("C:/TensorFlow Coursera/cats-v-dogs")
    os.mkdir("C:/TensorFlow Coursera/cats-v-dogs/training")
    os.mkdir("C:/TensorFlow Coursera/cats-v-dogs/training/cats")
    os.mkdir("C:/TensorFlow Coursera/cats-v-dogs/training/dogs/")
    os.mkdir("C:/TensorFlow Coursera/cats-v-dogs/testing")
    os.mkdir("C:/TensorFlow Coursera/cats-v-dogs/testing/cats")
    os.mkdir("C:/TensorFlow Coursera/cats-v-dogs/testing/dogs")
except OSError:
    print("Some Error happens!!")
    pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # YOUR CODE STARTS HERE
    all_images = os.listdir(SOURCE)
    shuffle(all_images)
    splitting_index = round(SPLIT_SIZE * len(all_images))
    train_images = all_images[:splitting_index]
    test_images = all_images[splitting_index:]
    # copy training images
    for img in train_images:
        src = os.path.join(SOURCE, img)
        dst = os.path.join(TRAINING, img)
        if os.path.getsize(src) <= 0:
            print(img + " is zero length, so ignoring!!")
        else:
            shutil.copyfile(src, dst)
    # copy testing images
    for img in test_images:
        src = os.path.join(SOURCE, img)
        dst = os.path.join(TESTING, img)
        if os.path.getsize(src) <= 0:
            print(img + " is zero length, so ignoring!!")
        else:
            shutil.copyfile(src, dst)


CAT_SOURCE_DIR = "C:/TensorFlow Coursera/PetImages/Cat/"
TRAINING_CATS_DIR = "C:/TensorFlow Coursera/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "C:/TensorFlow Coursera/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "C:/TensorFlow Coursera/PetImages/Dog/"
TRAINING_DOGS_DIR = "C:/TensorFlow Coursera/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "C:/TensorFlow Coursera/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir('C:/TensorFlow Coursera/cats-v-dogs/training/cats/')))
print(len(os.listdir('C:/TensorFlow Coursera/cats-v-dogs/training/dogs/')))
print(len(os.listdir('C:/TensorFlow Coursera/cats-v-dogs/testing/cats/')))
print(len(os.listdir('C:/TensorFlow Coursera/cats-v-dogs/testing/dogs/')))

# Expected output:
# 1350
# 1350
# 150
# 150

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = 'C:/TensorFlow Coursera/cats-v-dogs/training'
train_datagen = ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=10, target_size=(150, 150),
                                                    class_mode='binary')

VALIDATION_DIR = 'C:/TensorFlow Coursera/cats-v-dogs/testing'
validation_datagen = ImageDataGenerator(rescale=1. / 255.)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size=10, target_size=(150, 150),
                                                              class_mode='binary')

# Expected Output:
# Found 2700 images belonging to 2 classes.
# Found 300 images belonging to 2 classes.

history = model.fit_generator(train_generator,
                              epochs=2,
                              verbose=1,
                              validation_data=validation_generator)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.show()

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.title('Training and validation loss')
plt.show()
