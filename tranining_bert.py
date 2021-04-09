from shutil import copyfile
import re
import os
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
# copyfile(src = "../input/shopee-external-models/tokenization.py", dst = "../working/tokenization.py")
import tokenization
import tensorflow_hub as hub

# Configuration
EPOCHS = 25
BATCH_SIZE = 8
# Seed
SEED = 123
# Verbosity
VERBOSE = 1
LR = 0.00001


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def read_and_preprocess():
    df = pd.read_csv('./shopee-product-matching/train.csv')
    tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
    df['matches'] = df['label_group'].map(tmp)
    df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
    encoder = LabelEncoder()
    df['label_group'] = encoder.fit_transform(df['label_group'])
    N_CLASSES = df['label_group'].nunique()
    print(f'We have {N_CLASSES} classes')
    x_train, x_val, y_train, y_val = train_test_split(df[['title']], df['label_group'], shuffle=True,
                                                      stratify=df['label_group'], random_state=SEED, test_size=0.33)
    return df, N_CLASSES, x_train, x_val, y_train, y_val


# Return tokens, masks and segments from a text array or series
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


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


# Function to build bert model
def build_bert_model(bert_layer, max_len=512):
    margin = ArcMarginProduct(
        n_classes=N_CLASSES,
        s=30,
        m=0.5,
        name='head/arc_margin',
        dtype='float32'
    )

    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    label = tf.keras.layers.Input(shape=(), name='label')

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    x = margin([clf_output, label])
    output = tf.keras.layers.Softmax(dtype='float32')(x)
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids, label], outputs=[output])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR),
                  loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def load_train_and_evaluate(x_train, x_val, y_train, y_val):
    seed_everything(SEED)
    # Load BERT from the Tensorflow Hub
    module_url = "./bert_en_uncased_L-24_H-1024_A-16_1"
    bert_layer = hub.KerasLayer(module_url, trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    x_train = bert_encode(x_train['title'].values, tokenizer, max_len=70)
    x_val = bert_encode(x_val['title'].values, tokenizer, max_len=70)
    y_train = y_train.values
    y_val = y_val.values
    # Add targets to train and val
    x_train = (x_train[0], x_train[1], x_train[2], y_train)
    x_val = (x_val[0], x_val[1], x_val[2], y_val)
    bert_model = build_bert_model(bert_layer, max_len=70)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'Bert_{SEED}.h5',
                                                    monitor='val_loss',
                                                    verbose=VERBOSE,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='min')
    history = bert_model.fit(x_train, y_train,
                             validation_data=(x_val, y_val),
                             epochs=EPOCHS,
                             callbacks=[checkpoint],
                             batch_size=BATCH_SIZE,
                             verbose=VERBOSE)


df, N_CLASSES, x_train, x_val, y_train, y_val = read_and_preprocess()
load_train_and_evaluate(x_train, x_val, y_train, y_val)
