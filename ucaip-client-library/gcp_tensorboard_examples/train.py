from __future__ import print_function
import argparse
from datetime import datetime
import hypertune
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import sys
import os
import time

def get_args():
  """Argument parser.

    Returns:
      Dictionary of arguments.
    """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num-epochs',
      type=int,
      default=20,
      help='number of times to go through the data, default=20')
  parser.add_argument(
      '--batch-size',
      default=128,
      type=int,
      help='number of records to read during each training step, default=128')
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO')
  args, _ = parser.parse_known_args()
  return args


args = get_args()
tf.compat.v1.logging.set_verbosity(args.verbosity)
print('args', sys.argv)

batch_size = args.batch_size
num_classes = 10
epochs = args.num_epochs
print('epochs', epochs)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])

# Configuring TB.gcp integration
log_dir = os.environ['AIP_TENSORBOARD_LOG_DIR']
print("TB log dir: " + log_dir)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)
score = model.evaluate(x_test, y_test, verbose=0)
test_loss = score[0]
print('Test loss:', test_loss)
print('Test accuracy:', score[1])

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='test_loss',
    metric_value=test_loss,
    global_step=epochs-1)

# giving some time for hp paramerter tuning lib to read the test loss
time.sleep(30)
