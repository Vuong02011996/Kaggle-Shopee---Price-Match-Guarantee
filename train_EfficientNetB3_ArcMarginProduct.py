#!pip install -q efficientnet
#!pip install tensorflow_addons
# pip install kaggledatasets
import re
import os
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tqdm.notebook import tqdm
# from kaggle_datasets import KaggleDatasets

# Default distribution strategy in Tensorflow. Works on CPU and single GPU.
strategy = tf.distribute.get_strategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)

# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_PATH = './shopee-tf-records-512-stratified'

# Configuration
EPOCHS = 20
BATCH_SIZE = 4 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512, 512]
# Seed
SEED = 42
# Learning rate
LR = 0.001
# Verbosity
VERBOSE = 2
# Number of classes
N_CLASSES = 11011
# Number of folds
FOLDS = 5

# Training filenames directory
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/*.tfrec')


# Function to get our f1 score
def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def arcface_format(posting_id, image, label_group, matches):
    return posting_id, {'inp1': image, 'inp2': label_group}, label_group, matches


# Data augmentation function
def data_augment(posting_id, image, label_group, matches):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)
    return posting_id, image, label_group, matches


# Function to decode our images
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


# This function parse our images and also get the target variable
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "posting_id": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "label_group": tf.io.FixedLenFeature([], tf.int64),
        "matches": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    posting_id = example['posting_id']
    image = decode_image(example['image'])
#     label_group = tf.one_hot(tf.cast(example['label_group'], tf.int32), depth = N_CLASSES)
    label_group = tf.cast(example['label_group'], tf.int32)
    matches = example['matches']
    return posting_id, image, label_group, matches


# This function loads TF Records and parse them into tensors
def load_dataset(filenames, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    return dataset


# This function is to get our training tensors
def get_training_dataset(filenames, ordered = False):
    dataset = load_dataset(filenames, ordered = ordered)
    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
    dataset = dataset.map(arcface_format, num_parallel_calls = AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


# This function is to get our validation tensors
def get_validation_dataset(filenames, ordered = True):
    dataset = load_dataset(filenames, ordered = ordered)
    dataset = dataset.map(arcface_format, num_parallel_calls = AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


# Function to count how many photos we have in
def count_data_items(filenames):
    # The number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
print(f'Dataset: {NUM_TRAINING_IMAGES} training images')


# Function for a custom learning rate scheduler with warmup and decay
def get_lr_callback():
    lr_start = 0.000001
    lr_max = 0.000005 * BATCH_SIZE
    lr_min = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback


# Arcmarginproduct class keras layer
class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# Function to create our EfficientNetB3 model
def get_model():
    with strategy.scope():
        margin = ArcMarginProduct(
            n_classes=N_CLASSES,
            s=30,
            m=0.5,
            name='head/arc_margin',
            dtype='float32'
        )

        inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')
        label = tf.keras.layers.Input(shape=(), name='inp2')
        x = efn.EfficientNetB3(weights='imagenet', include_top=False)(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = margin([x, label])

        output = tf.keras.layers.Softmax(dtype='float32')(x)

        model = tf.keras.models.Model(inputs=[inp, label], outputs=[output])

        opt = tf.keras.optimizers.Adam(learning_rate=LR)

        model.compile(
            optimizer=opt,
            loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        return model


def train_and_evaluate():
    # Seed everything
    seed_everything(SEED)

    print('\n')
    print('-' * 50)
    train, valid = train_test_split(TRAINING_FILENAMES, shuffle=True, random_state=SEED)
    train_dataset = get_training_dataset(train, ordered=False)
    train_dataset = train_dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))
    val_dataset = get_validation_dataset(valid, ordered=True)
    val_dataset = val_dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))
    STEPS_PER_EPOCH = count_data_items(train) // BATCH_SIZE
    K.clear_session()
    model = get_model()
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'EfficientNetB3_{IMAGE_SIZE[0]}_{SEED}.h5',
                                                    monitor='val_loss',
                                                    verbose=VERBOSE,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='min')

    history = model.fit(train_dataset,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        callbacks=[checkpoint, get_lr_callback()],
                        validation_data=val_dataset,
                        verbose=VERBOSE)


train_and_evaluate()