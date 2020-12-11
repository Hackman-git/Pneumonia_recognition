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


IMAGE_SIZE = (180,180)
AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 25
INPUT_TENSOR_NAME = 'inputs_input'

def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()])
    
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
        
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', name='inputs',input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
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
        opt = SGD(lr=learning_rate , decay=weight_decay, momentum=momentum)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=learning_rate , decay=weight_decay)
    else:
        opt = Adam(lr=learning_rate, decay=weight_decay)
    

    METRICS = ['accuracy', tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
    
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=METRICS)
    
    return model

def _input(epochs, batch_size, channel, channel_name, shuffle_buffer_size=1000):
    obj = _get_datasets(channel, channel_name)
    dataset = obj[0]
    TRAIN_IMG_COUNT,VAL_IMG_COUNT, TEST_IMAGE_COUNT, class_weight = obj[1],obj[2], obj[3],obj[4]
    
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat infinitely.
    dataset = dataset.repeat()
    
    dataset = dataset.prefetch(10)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    
    image_batch, label_batch = iterator.get_next()
    

    return ({INPUT_TENSOR_NAME: image_batch}, label_batch, TRAIN_IMG_COUNT, VAL_IMG_COUNT,TEST_IMAGE_COUNT, class_weight)

def train_input_fn():
    return _input(args.epochs, args.batch_size, args.train, 'train')

def eval_input_fn():
    return _input(args.epochs, args.batch_size, args.eval, 'eval')

def validation_input_fn():
    return _input(args.epochs, args.batch_size, args.validation, 'validation')


def _get_datasets(channel, channel_name):
    filenames = tf.io.gfile.glob(os.path.join(args.train, '*/*'))
    filenames.extend(tf.io.gfile.glob(os.path.join(args.validation, '*/*')))
    train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)
    
    train_normal_count = len([filename for filename in train_filenames if "NORMAL" in filename])
    train_abnormal_count = len([filename for filename in train_filenames if "PNEUMONIA" in filename])
    
    train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
    val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)
    
    train_ds = train_list_ds.map(_process_path)
    val_ds = val_list_ds.map(_process_path)
    
    TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
    VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
    
     
    test_filenames = tf.io.gfile.glob(os.path.join(args.eval, '*/*'))
    test_list_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
    TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
    test_ds = test_list_ds.map(_process_path)
    
    weight_for_0 = (1 / train_normal_count)*(TRAIN_IMG_COUNT)/2.0 
    weight_for_1 = (1 / train_abnormal_count)*(TRAIN_IMG_COUNT)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print('in get_ds function: '+str(TRAIN_IMG_COUNT)+' '+str(VAL_IMG_COUNT)+' '+str(TEST_IMAGE_COUNT), flush=True)
    
    if channel_name == 'train':
        return (train_ds,TRAIN_IMG_COUNT,VAL_IMG_COUNT, TEST_IMAGE_COUNT, class_weight)
    if channel_name == 'validation':
        return (val_ds,TRAIN_IMG_COUNT,VAL_IMG_COUNT,TEST_IMAGE_COUNT, class_weight)
    if channel_name == 'eval':
        return (test_ds,TRAIN_IMG_COUNT,VAL_IMG_COUNT,TEST_IMAGE_COUNT, class_weight)
    
    
    
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

def save_model(model, output):
    
    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.saved_model.save(model, output+'/1/') 
    logging.info("Model successfully saved at: {}".format(output))
    return


def main(args):
    logging.info("getting data")
    obj = train_input_fn()
    train_dataset = obj[0:2]
    TRAIN_IMG_COUNT,VAL_IMG_COUNT,TEST_IMG_COUNT,class_weight = obj[2:]
    
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
        train_dataset[0],
        train_dataset[1],
        steps_per_epoch=TRAIN_IMG_COUNT // args.batch_size,
        epochs=args.epochs,
        validation_data=validation_dataset,
        validation_steps=VAL_IMG_COUNT // args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks)


    score = model.evaluate(eval_dataset[0],
                           eval_dataset[1],
                        steps= TEST_IMG_COUNT // args.batch_size,
                        verbose=0)

    logging.info('Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))
    logging.info('Test precision:{}'.format(score[2]))
    logging.info('Test recall:{}'.format(score[3]))
    
    return save_model(model, args.model_output_dir)

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The directory where the input data is stored.')
    parser.add_argument(
        '--validation',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_VALIDATION'),
        help='The directory where the input data is stored.')
    parser.add_argument(
        '--eval',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_EVAL'),
        help='The directory where the input data is stored.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR'))
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
        default=1,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
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
    
    args = parser.parse_known_args()[0]
    main(args)