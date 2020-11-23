from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import re
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split


def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()]
    )
    
    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block

def keras_model_fn(learning_rate, weight_decay, optimizer):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model is transformed into a TensorFlow Estimator before training and saved in a
    TensorFlow Serving SavedModel at the end of training.
    """
    model = Sequential([
        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    if optimizer.lower() == 'sgd':
        opt = SGD(lr=learning_rate * size, decay=weight_decay, momentum=momentum)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=learning_rate * size, decay=weight_decay)
    else:
        opt = Adam(lr=learning_rate * size, decay=weight_decay)
    

    METRICS = ['accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')]
    
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=METRICS)
    
    return model

def _input(epochs, batch_size, channel_name, shuffle_buffer_size=1000):
    obj = _get_datasets(channel_name)
    dataset = obj[0]
    TRAIN_IMG_COUNT,VAL_IMG_COUNT, TEST_IMAGE_COUNT = obj[1],obj[2], obj[3]
    
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat infinitely.
    dataset = dataset.repeat()
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    image_batch, label_batch = next(iter(dataset))
    

    return ({INPUT_TENSOR_NAME: image_batch}, label_batch, TRAIN_IMG_COUNT, VAL_IMG_COUNT,TEST_IMAGE_COUNT)

def train_input_fn():
    return _input(args.epochs, args.batch_size, 'train')

def eval_input_fn():
    return _input(args.epochs, args.batch_size, 'eval')

def validation_input_fn():
    return _input(args.epochs, args.batch_size, 'validation')

def _prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def _get_datasets(channel_name):
    filenames = tf.io.gfile.glob(r'./chest_xray/train/*/*')
    filenames.extend(tf.io.gfile.glob(r'./chest_xray/val/*/*'))

    train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)
    
    train_normal_count = len([filename for filename in train_filenames if "NORMAL" in filename])
    train_abnormal_count = len([filename for filename in train_filenames if "PNEUMONIA" in filename])
    
    train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
    val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)
    
    train_ds = train_list_ds.map(_process_path)
    val_ds = val_list_ds.map(_process_path)
    
    TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
    VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
    
    test_list_ds = tf.data.Dataset.list_files(r'./chest_xray/test/*/*')
    TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
    test_ds = test_list_ds.map(_process_path)
    test_ds = test_ds.batch(BATCH_SIZE)
    
    if channel_name == 'train':
        return (train_ds,TRAIN_IMG_COUNT,VAL_IMG_COUNT, TEST_IMAGE_COUNT)
    elif channel_name == 'validation':
        return (val_ds,TRAIN_IMG_COUNT,VAL_IMG_COUNT,TEST_IMAGE_COUNT)
    else:
        return (test_ds,TRAIN_IMG_COUNT,VAL_IMG_COUNT,TEST_IMAGE_COUNT)
    
    
    
def _get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return int(parts[-2] == "PNEUMONIA")


def _decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, IMAGE_SIZE)


def _process_path(file_path):
    label = _get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = _decode_img(img)
    return img, label


def main(args):
    logging.info("getting data")
    obj = train_input_fn()
    train_dataset = obj[0:2]
    TRAIN_IMG_COUNT,VAL_IMG_COUNT, TEST_IMG_COUNT = obj[2:]
    eval_dataset = eval_input_fn()[0:2]
    validation_dataset = validation_input_fn()[0:2]
    

    logging.info("configuring model")
    model = keras_model_fn(args.learning_rate, args.weight_decay, args.optimizer)

    callbacks = []
    callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
    callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.h5'))

    logging.info("Starting training")
    size = 1
    
    model.fit(
        train_dataset,
        steps_per_epoch=TRAIN_IMG_COUNT // args.batch_size,
        epochs=args.epochs,
        validation_data=validation_dataset,
        validation_steps=VAL_IMG_COUNT // args.batch_size,
        class_weight=args.class_weight,
        callbacks=callbacks)


    score = model.evaluate(eval_dataset,
            steps= TEST_IMG_COUNT // args.batch_size,
            verbose=0)

    logging.info('Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))
    logging.info('Test precision:{}'.format(score[2]))
    logging.info('Test recall:{}'.format(score[3]))

    
if __name__ == '__main__':
    aws s3 cp s3://sagemaker-pneumonia-bucket/chest_xray ./chest_xray --recursive;
        
    IMAGE_SIZE = (180,180)
    BATCH_SIZE = 128
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    EPOCHS = 25
    INPUT_TENSOR_NAME = 'inputs_input'
    
    import json
    filenames = tf.io.gfile.glob(r'./chest_xray/train/*/*')
    filenames.extend(tf.io.gfile.glob(r'./chest_xray/val/*/*'))
    train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)
    train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
    TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
    train_normal_count = len([filename for filename in train_filenames if "NORMAL" in filename])
    train_abnormal_count = len([filename for filename in train_filenames if "PNEUMONIA" in filename])
    weight_for_0 = (1 / train_normal_count)*(TRAIN_IMG_COUNT)/2.0 
    weight_for_1 = (1 / train_abnormal_count)*(TRAIN_IMG_COUNT)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default= './')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size for training.')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default='0.9'
    )
    parser.add_argument(
        '--class_weight',
        type = json.loads,
        default=class_weight
    )
    
    args = parser.parse_known_args()[0]
    main(args)