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
AUTOTUNE = tf.data.experimental.AUTOTUNE
LOCAL_LOG_DIR = '/tmp/logs'

    
def build_model(num_layers, dropout_ratio, num_classes):
    """
    Creates a custom image classificatin model using ResNet50 
    as a base model.
    """
    
    # Create the base model
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet',
                                                   pooling='avg')
    base_model.trainable = False
    
    # Add preprocessing and classification head
    inputs = tf.keras.Input(shape=IMG_SHAPE, name='images', dtype = tf.uint8)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x)
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


def image_dataset_from_aip_jsonl(pattern, class_names=None, img_height=224, img_width=224):
        """
        Generates a `tf.data.Dataset` from a set of JSONL files
        in the AI Platform image dataset index format. 
        
        Arguments:
            pattern: A wildcard pattern for a list of JSONL files.
                E.g. gs://bucket/folder/training-*.
            class_names: the list of class names that are expected
                in the passed index. 
            img_height: The height of a generated image
            img_width: The width of a generated image
            
        """
        
        def _get_label(class_name):
            """
            Converts a string class name to an integer label.
            """
            one_hot = class_name == class_names
            return tf.argmax(one_hot)
        
        def _decode_img(file_path):
            """
            Loads an image and converts it to a resized 3D tensor.
            """
            
            img = tf.io.read_file(file_path)
            img = tf.io.decode_image(img, 
                                     expand_animations=False)
            img = tf.image.resize(img, [img_height, img_width])
            img = tf.cast(img, tf.uint8)
            
            return img
            
        def _process_example(file_path, class_name):
            """
            Creates a converted image and a class label from
            an image path and class name.
            """
            
            label = _get_label(class_name)
            img = _decode_img(file_path)
            
            return img, label
        
        # Read the JSONL index to a pandas DataFrame
        df = pd.concat(
            [pd.read_json(path, lines=True) for path in tf.io.gfile.glob(pattern)],
            ignore_index=True
        )
        
        # Parse classifcationAnnotations field
        df = pd.concat(
            [df, pd.json_normalize(df['classificationAnnotations'].apply(pd.Series)[0])], axis=1)
        
        paths = df['imageGcsUri'].values
        labels = df['displayName'].values
        inferred_class_names = np.unique(labels)
        
        if class_names is not None:
            class_names = np.array(class_names).astype(str)
            if set(inferred_class_names) != set(class_names):
                raise ValueError(
                    'The `class_names` passed do not match the '
                    'names in the image index '
                    'Expected: %s, received %s' %
                    (inferred_class_names, class_names))
            
        class_names = tf.constant(inferred_class_names)
        
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.shuffle(len(labels), reshuffle_each_iteration=False)
        dataset = dataset.map(_process_example, num_parallel_calls=AUTOTUNE)
        dataset.class_names = class_names
        
        return dataset


def get_datasets(batch_size, img_height, img_width):
    """
    Creates training and validation splits as tf.data datasets
    from an AI Platform Dataset passed by the training pipeline.
    """
    
    def _configure_for_performance(ds):
        """
        Optimizes the performance of a dataset.
        """
        ds = ds.cache()
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    
    if  os.environ['AIP_DATA_FORMAT'] != 'jsonl':
        raise RuntimeError('Wrong dataset format: {}. Expecting - jsonl'.format(
            os.environ['AIP_DATA_FORMAT']))
        
    train_ds = image_dataset_from_aip_jsonl(
        pattern=os.environ['AIP_TRAINING_DATA_URI'],
        img_height=img_height,
        img_width=img_width)
    
    class_names = train_ds.class_names.numpy()
        
    valid_ds = image_dataset_from_aip_jsonl(
        pattern=os.environ['AIP_VALIDATION_DATA_URI'],
        class_names=class_names,
        img_height=img_height,
        img_width=img_width)
    
    train_ds = _configure_for_performance(train_ds.batch(batch_size))
    valid_ds = _configure_for_performance(valid_ds.batch(batch_size))
    
    return train_ds, valid_ds, class_names
        
    
def get_args():
    """
    Returns parsed command line arguments.
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


def copy_tensorboard_logs(local_path: str, gcs_path: str):
    """Copies Tensorboard logs from a local dir to a GCS location.
    
    After training, batch copy Tensorboard logs locally to a GCS location. This can result
    in faster pipeline runtimes over streaming logs per batch to GCS that can get bottlenecked
    when streaming large volumes.
    
    Args:
      local_path: local filesystem directory uri.
      gcs_path: cloud filesystem directory uri.
    Returns:
      None.
    """
    pattern = '{}/*/events.out.tfevents.*'.format(local_path)
    local_files = tf.io.gfile.glob(pattern)
    gcs_log_files = [local_file.replace(local_path, gcs_path) for local_file in local_files]
    for local_file, gcs_file in zip(local_files, gcs_log_files):
        tf.io.gfile.copy(local_file, gcs_file)


if __name__ == '__main__':
    
    if 'AIP_DATA_FORMAT' not in os.environ:
        raise RuntimeError('No dataset information available.')
        
    # Configure TensorBoard callback
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=LOCAL_LOG_DIR, update_freq='batch')]
   
    args = get_args()
    
    # Create the datasets and the model
    train_ds, valid_ds, class_names = get_datasets(args.batch_size, IMG_HEIGHT, IMG_WIDTH)
    model = build_model(args.num_layers, args.dropout_ratio, len(class_names))
    print(model.summary())
    
    # Start training
    history = model.fit(x=train_ds, 
                        validation_data=valid_ds, 
                        epochs=args.num_epochs,
                        callbacks=callbacks)
    
    # Configure locations for SavedModel and TB logs
    if 'AIP_MODEL_DIR' in os.environ:
        model_dir = os.environ['AIP_MODEL_DIR']
    else:
        model_dir = args.model_dir
    tb_dir = f'{model_dir}/logs'
    
    # Save the model
    print('Saving the model to: {}'.format(model_dir))
    model.save(model_dir)
    
    # Copy Tensorboard logs to GCS
    copy_tensorboard_logs(LOCAL_LOG_DIR, tb_dir)
