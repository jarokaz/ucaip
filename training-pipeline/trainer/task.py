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

import argparse
import os
import time

#import hypertune
import numpy as np
import pandas as pd
import tensorflow as tf


IMG_HEIGHT = 224
IMG_WIDTH = 224
    
    
def build_model(num_layers, dropout_ratio, num_classes):
    """
    Creates a custom image classificatin model using ResNet50 
    as a base model.
    """
    
    # Create the base model
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet',
                                                   pooling='avg')
    base_model.trainable = False
    
    # Add preprocessing and classification head
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.Dense(num_layers, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_ratio)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    
    # Assemble the model
    model = tf.keras.Model(inputs, outputs)
    
    # Compile the model
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model


def get_datasets(batch_size):
    """
    Creates training and validation splits as 
    tf.data datasets.
    """
    
    url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    data_root = tf.keras.utils.get_file('flower_photos', origin=url, untar=True)
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)
    
    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)
    
    class_names = train_ds.class_names
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, valid_ds, class_names

    
def get_args():
    """
    Returns parsed command line arguments
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int,
        help='number of records to read during each training step, default=32')
    parser.add_argument(
        '--num-layers',
        default=64,
        type=int,
        help='number of hidden layers in the classification head , default=64')
    parser.add_argument(
        '--dropout-ratio',
        default=0.5,
        type=float,
        help='dropout ration in the classification head , default=128')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/saved_model',
        help='model dir , default=/tmp/saved_model')

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    
   
    args = get_args()
                  
    # Check for GPU and set the strategy
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    
    # Create the datasets and the model
    train_ds, valid_ds, class_names = get_datasets(args.batch_size)
    with strategy.scope():
        model = build_model(args.num_layers, args.dropout_ratio, len(class_names))
    print(model.summary())
    
    # Start training
    history = model.fit(x=train_ds, 
                        validation_data=valid_ds, 
                        epochs=args.num_epochs)
    model.save(args.model_dir)

