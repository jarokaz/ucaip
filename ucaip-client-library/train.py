
# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from datetime import datetime
import hypertune
import tensorflow as tf
import sys
import os
import time



def _get_model(input_shape, num_classes):
    """
    Creates a simple convolutional network.
    """
    
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=['accuracy'])
    
    return model
  

def _get_datasets():
    """
    Creates MNIST training and validation splits.
    """
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    img_rows, img_cols = 28, 28
    num_classes = 10
    
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    return input_shape, num_classes, x_train, y_train, x_test, y_test


class _HptuneCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback class that reports a metric to hypertuner
    at the end of each epoch.
    """
    
    def __init__(self, metric_tag, metric_value):
        super(_HptuneCallback, self).__init__()
        self.metric_tag = metric_tag
        self.metric_value = metric_value
        self.hpt = hypertune.HyperTune()
        
    def on_epoch_end(self, epoch, logs=None):
        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.metric_tag,
            metric_value=logs[self.metric_value],
            global_step=epoch)


def train(batch_size, epochs, verbosity):
    """
    Trains the mnist model.
    """
    
    # Prepare datasets
    input_shape, num_classes, x_train, y_train, x_test, y_test = _get_datasets()
    
    # Create model
    model = _get_model(input_shape, num_classes)
    
    # Configure Hypertuner callback
    callbacks = [_HptuneCallback('accuracy', 'val_accuracy')]
            
    # Configure TensorBoard callback
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        log_dir = os.environ['AIP_TENSORBOARD_LOG_DIR']
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    
    # Start training
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )
    

    
def get_args():
  """
  Returns an argument parser.
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



if __name__ == "__main__":
    args = get_args()
    train(args.batch_size, args.num_epochs, args.verbosity)
