# Install RAPIDS
# !git clone https://github.com/rapidsai/rapidsai-csp-utils.git
# !bash rapidsai-csp-utils/colab/rapids-colab.sh stable
#
# import sys, os
#
# dist_package_index = sys.path.index('/usr/local/lib/python3.7/dist-packages')
# sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.7/site-packages'] + sys.path[dist_package_index:]
# sys.path
# exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())

import numpy as np, pandas as pd, gc
import cv2, matplotlib.pyplot as plt
import cudf, cuml, cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
print('RAPIDS', cuml.__version__)
print('TF', tf.__version__)


def load_train_data():
    COMPUTE_CV = True

    test = pd.read_csv('../input/shopee-product-matching/test.csv')
    print(len(test))
    if len(test) > 3:
        COMPUTE_CV = False
    else:
        print('this submission notebook will compute CV score, but commit notebook will not')

    train = pd.read_csv('../input/shopee-product-matching/train.csv')
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    print('train shape is', train.shape)
    train.head()


def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score


def compute_baseline_cv_score(train):
    """
    A baseline is to predict all items with the same image_phash as being duplicate.
    Let's calcuate the CV score for this submission.
    :return:
    """

    tmp = train.groupby('image_phash').posting_id.agg('unique').to_dict()
    train['oof'] = train.image_phash.map(tmp)

    train['f1'] = train.apply(getMetric('oof'), axis=1)
    print('CV score for baseline =', train.f1.mean())


def compute_rapids_model_cv_and_infer_submission():
    if COMPUTE_CV:
        test = pd.read_csv('../input/shopee-product-matching/train.csv')
        test_gf = cudf.DataFrame(test)
        print('Using train as test to compute CV (since commit notebook). Shape is', test_gf.shape)
    else:
        test = pd.read_csv('../input/shopee-product-matching/test.csv')
        test_gf = cudf.read_csv('../input/shopee-product-matching/test.csv')
        print('Test shape is', test_gf.shape)
    test_gf.head()


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, df, img_size=256, batch_size=32, path=''):
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        self.path = path
        self.indexes = np.arange(len(self.df))

    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = len(self.df) // self.batch_size
        ct += int(((len(self.df)) % self.batch_size) != 0)
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(indexes)
        return X

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        X = np.zeros((len(indexes), self.img_size, self.img_size, 3), dtype='float32')
        df = self.df.iloc[indexes]
        for i, (index, row) in enumerate(df.iterrows()):
            img = cv2.imread(self.path + row.image)
            X[i,] = cv2.resize(img, (self.img_size, self.img_size))  # /128.0 - 1.0
        return X


def image_embeddings_extraction(test):
    BASE = '../input/shopee-product-matching/test_images/'
    if COMPUTE_CV:
        BASE = '../input/shopee-product-matching/train_images/'

    WGT = '../input/effnetb0/efficientnetb0_notop.h5'
    model = EfficientNetB0(weights=WGT, include_top=False, pooling='avg', input_shape=None)

    embeds = []
    CHUNK = 1024 * 4

    print('Computing image embeddings...')
    CTS = len(test) // CHUNK
    if len(test) % CHUNK != 0:
        CTS += 1
    for i, j in enumerate(range(CTS)):
        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(test))
        print('chunk', a, 'to', b)

        test_gen = DataGenerator(test.iloc[a:b], batch_size=32, path=BASE)
        image_embeddings = model.predict(test_gen, verbose=1, use_multiprocessing=True, workers=4)
        embeds.append(image_embeddings)

        # if i>=1: break

    del model
    _ = gc.collect()
    image_embeddings = np.concatenate(embeds)
    print('image embeddings shape', image_embeddings.shape)


def find_similar_image():
    KNN = 50
    if len(test) == 3:
        KNN = 2

    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(image_embeddings)
    preds = []
    CHUNK = 1024 * 4

    print('Finding similar images...')
    CTS = len(image_embeddings) // CHUNK
    if len(image_embeddings) % CHUNK != 0:
        CTS += 1
    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(image_embeddings))
        print('chunk', a, 'to', b)
        distances, indices = model.kneighbors(image_embeddings[a:b, ])

        for k in range(b - a):
            IDX = np.where(distances[k,] < 6.0)[0]
            IDS = indices[k, IDX]
            o = test.iloc[IDS].posting_id.values
            preds.append(o)

    del model, distances, indices, image_embeddings, embeds
    _ = gc.collect()

    test['preds2'] = preds
    test.head()


def text_embedding_extraction():
    print('Computing text embeddings...')
    model = TfidfVectorizer(stop_words='english', binary=True, max_features=25_000)
    text_embeddings = model.fit_transform(test_gf.title).toarray()
    print('text embeddings shape', text_embeddings.shape)


def find_similar_titles():
    preds = []
    CHUNK = 1024 * 4

    print('Finding similar titles...')
    CTS = len(test) // CHUNK
    if len(test) % CHUNK != 0: CTS += 1
    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(test))
        print('chunk', a, 'to', b)

        # COSINE SIMILARITY DISTANCE
        cts = cupy.matmul(text_embeddings, text_embeddings[a:b].T).T

        for k in range(b - a):
            IDX = cupy.where(cts[k,] > 0.7)[0]
            o = test.iloc[cupy.asnumpy(IDX)].posting_id.values
            preds.append(o)

    del model, text_embeddings
    _ = gc.collect()

    test['preds'] = preds
    test.head()


def find_similar_phash_feature():
    tmp = test.groupby('image_phash').posting_id.agg('unique').to_dict()
    test['preds3'] = test.image_phash.map(tmp)
    test.head()

# Compute CV Score

def combine_for_sub(row):
    x = np.concatenate([row.preds,row.preds2, row.preds3])
    return ' '.join( np.unique(x) )

def combine_for_cv(row):
    x = np.concatenate([row.preds,row.preds2, row.preds3])
    return np.unique(x)

def compute_cv_score():
    if COMPUTE_CV:
        tmp = test.groupby('label_group').posting_id.agg('unique').to_dict()
        test['target'] = test.label_group.map(tmp)
        test['oof'] = test.apply(combine_for_cv, axis=1)
        test['f1'] = test.apply(getMetric('oof'), axis=1)
        print('CV Score =', test.f1.mean())

    test['matches'] = test.apply(combine_for_sub, axis=1)


def write_submit_csv():
    test[['posting_id', 'matches']].to_csv('submission.csv', index=False)
    sub = pd.read_csv('submission.csv')
    sub.head()