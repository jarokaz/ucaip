# Custom Job for CIFAR10

import tensorflow_datasets as tfds
import tensorflow as tf
from hypertune import HyperTune
import argparse
import os
import sys

# Command Line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--job-dir')
parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir.')
parser.add_argument('--lr', dest='lr',
                    default=0.001, type=float,
                    help='Learning rate.')
parser.add_argument('--decay', dest='decay',
                    default=0.01, type=float,
                    help='Decay rate')
args = parser.parse_args()

# Scaling CIFAR-10 data from (0, 255] to (0., 1.]
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label

# Download the dataset
datasets = tfds.load(name='cifar10', as_supervised=True)

# Preparing dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = datasets['train'].map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = datasets['test'].map(scale).batch(BATCH_SIZE)

# Build the Keras model
def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr, decay=args.decay),
      metrics=['accuracy'])
  return model

model = build_and_compile_cnn_model()

# Instantiate the HyperTune reporting object
hpt = HyperTune()

# Reporting callback
class HPTCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        global hpt
        hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='val_accuracy',
        metric_value=logs['val_accuracy'],
        global_step=epoch)

# Train the model
model.fit(train_dataset, epochs=5, steps_per_epoch=10, validation_data=test_dataset.take(8),
    callbacks=[HPTCallback()])
model.save(args.model_dir)
