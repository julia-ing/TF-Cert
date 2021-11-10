import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.optimizer_v1 import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras import Model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
C2 imageGenerator / C3 transfer-learning / CNN, data-augmentation 꼭 해주기
"""

# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#             -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

path_inception = '/Users/yewon/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
local_weights_file = path_inception

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')  # 7x7
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

model.summary()

# 코랩에서 다운받는 법
# !gdown --id 1onaG42NZft3wCE1WH0GDEbUhu75fedP5
# !gdown --id 1LYeusSEIiZQpwN-mthh5nKdA75VsKG1U


"""
만약 zip 파일로 제공되는 경우
"""
# import zipfile

# test_local_zip = './horse-or-human.zip'
# zip_ref = zipfile.ZipFile(test_local_zip, 'r')
# zip_ref.extractall('./training')
#
# val_local_zip = './validation-horse-or-human.zip'
# zip_ref = zipfile.ZipFile(val_local_zip, 'r')
# zip_ref.extractall('./validation')
#
# zip_ref.close()

train_dir = './horse-or-human'
validation_dir = './validation-horse-or-human'

train_horses_dir = os.path.join(train_dir, 'horses')
train_humans_dir = os.path.join(train_dir, 'humans')
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_humans_fnames))
print(len(validation_humans_fnames))

train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

callbacks = myCallback()
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_steps=5,
    verbose=2,
    callbacks=callbacks)

"""
CNN 모델
"""

cnn_model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[150, 150, 3]),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Dropout(0.5),

        # Neural Network Building
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),  # Input Layer
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=256, activation='relu'),  # Hidden Layer
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(units=2, activation='softmax'),  # Output Layer
    ]
)

cnn_model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

cnn_model.compile(
    optimizer=Adam(lr=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
