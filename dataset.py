import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
print('TF',tf.__version__)

import cudf, cuml, cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
print('RAPIDS',cuml.__version__)


def restrict_tensorflow_mem_with_gpu_ram():
    # SO THAT WE HAVE GPU RAM FOR RAPIDS CUML KNN
    LIMIT = 12
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * LIMIT)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    print('Restrict TensorFlow to max %iGB GPU RAM' % LIMIT)
    print('so RAPIDS can use %iGB GPU RAM' % (16 - LIMIT))


def load_train_data():
    train = pd.read_csv('./shopee-product-matching/train.csv')
    print('train shape is', train.shape)
    train.head()

    return train


BASE = './shopee-product-matching/train_images/'


def displayDF(train, random=False, COLS=6, ROWS=4, path=BASE):
    for k in range(ROWS):
        plt.figure(figsize=(20, 5))
        for j in range(COLS):
            if random:
                row = np.random.randint(0, len(train))
            else:
                row = COLS * k + j
            name = train.iloc[row, 1]
            title = train.iloc[row, 3]
            title_with_return = ""
            for i, ch in enumerate(title):
                title_with_return += ch
                if (i != 0) & (i % 20 == 0): title_with_return += '\n'
            img = cv2.imread(path + name)
            plt.subplot(1, COLS, j + 1)
            plt.title(title_with_return)
            plt.axis('off')
            # plt.imshow(img)
            plt.imshow(img[:, :, ::-1])
    plt.show()


def display_duplicated_items():
    groups = train.label_group.value_counts()
    print(groups.shape)
    plt.figure(figsize=(20, 5))
    plt.plot(np.arange(len(groups)), groups.values)
    plt.ylabel('Duplicate Count', size=14)
    plt.xlabel('Index of Unique Item', size=14)
    plt.title('Duplicate Count vs. Unique Item Count', size=16)
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.bar(groups.index.values[:50].astype('str'), groups.values[:50])
    plt.xticks(rotation=45)
    plt.ylabel('Duplicate Count', size=14)
    plt.xlabel('Label Group', size=14)
    plt.title('Top 50 Duplicated Items', size=16)
    plt.show()

    # TOP 5 duplicated
    for k in range(5):
        print('#' * 40)
        print('### TOP %i DUPLICATED ITEM:' % (k + 1), groups.index[k])
        print('#' * 40)
        top = train.loc[train.label_group == groups.index[k]]

        displayDF(top, random=False, ROWS=2, COLS=4)


def cv2_view(path=BASE):
    groups = train.label_group.value_counts()
    print(groups.shape)
    size_image = 50
    num_col = 20
    # TOP 5 duplicated
    for k in range(5):
        print('#' * 40)
        print('### TOP %i DUPLICATED ITEM:' % (k + 1), groups.index[k])
        print('#' * 40)
        top = train.loc[train.label_group == groups.index[k]]
        image_name_list = top.image.values

        num_img = 0
        combine_image = None
        num_row = 0

        sample = cv2.imread(path + image_name_list[0])
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        sample = cv2.resize(sample, (size_image, size_image), interpolation=cv2.INTER_AREA)

        for image_name in image_name_list[1:]:
            arr_image = cv2.imread(path + image_name)
            arr_image = cv2.cvtColor(arr_image, cv2.COLOR_BGR2RGB)
            arr_image = cv2.resize(arr_image, (size_image, size_image), interpolation=cv2.INTER_AREA)

            num_img += 1
            if num_img % num_col == 0:
                if combine_image is None:
                    combine_image = sample
                else:
                    combine_image = np.concatenate((combine_image, sample), axis=0)
                sample = arr_image
                num_row += 1
            else:
                sample = np.concatenate((sample, arr_image), axis=1)
            # cv2.imshow("window", arr_image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            print("num_img", num_img)
        plt.figure(figsize=(6 * 3.13, 4 * 3.13))
        plt.imshow(combine_image)
        plt.show()
        # while True:
        #     cv2.imshow("window", combine_image)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break


def find_similar_titles_with_rapids_knn():
    """
    First we will extract text embeddings using RAPIDS cuML's TfidfVectorizer.
    This will turn every title into a one-hot-encoding of the words present.
    We will then compare one-hot-encodings with RAPIDS cuML KNN to find title's that are similar.
    :return:
    """
    # LOAD TRAIN UNTO THE GPU WITH CUDF
    train_gf = cudf.read_csv('../input/shopee-product-matching/train.csv')
    print('train shape is', train_gf.shape)
    train_gf.head()

    # Extract Text Embeddings with RAPIDS TfidfVectorizer¶
    # TfidfVectorizer returns a cupy sparse matrix.
    # Afterward we convert to a cupy dense matrix and feed that into RAPIDS cuML KNN.

    model = TfidfVectorizer(stop_words='english', binary=True)
    text_embeddings = model.fit_transform(train_gf.title).toarray()
    print('text embeddings shape is', text_embeddings.shape)

    # After fitting KNN, we will display some example rows of train and their 10 closest other titles in train (based on word count one-hot-encoding).

    KNN = 50
    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(text_embeddings)
    distances, indices = model.kneighbors(text_embeddings)

    for k in range(5):
        plt.figure(figsize=(20, 3))
        plt.plot(np.arange(50), cupy.asnumpy(distances[k,]), 'o-')
        plt.title('Text Distance From Train Row %i to Other Train Rows' % k, size=16)
        plt.ylabel('Distance to Train Row %i' % k, size=14)
        plt.xlabel('Index Sorted by Distance to Train Row %i' % k, size=14)
        plt.show()

        print(train_gf.loc[cupy.asnumpy(indices[k, :10]), ['title', 'label_group']])


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, img_size=256, batch_size=32, path=BASE):
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        self.path = path
        self.indexes = np.arange( len(self.df) )

    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = len(self.df) // self.batch_size
        ct += int(( (len(self.df)) % self.batch_size)!=0)
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(indexes)
        return X

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        X = np.zeros((len(indexes),self.img_size,self.img_size,3),dtype='float32')
        df = self.df.iloc[indexes]
        for i,(index,row) in enumerate(df.iterrows()):
            img = cv2.imread(self.path+row.image)
            X[i,] = cv2.resize(img,(self.img_size,self.img_size)) #/128.0 - 1.0
        return X


def find_matching_image_with_rapids():

    model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=None)
    train_gen = DataGenerator(train, batch_size=128)
    image_embeddings = model.predict(train_gen, verbose=1)
    print('image embeddings shape is', image_embeddings.shape)

    # After fitting KNN, we will display some example rows of train and their 8 closest other images in train (based EffNetB0 image embeddings).

    KNN = 50
    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(image_embeddings)
    distances, indices = model.kneighbors(image_embeddings)

    for k in range(180, 190):
        plt.figure(figsize=(20, 3))
        plt.plot(np.arange(50), cupy.asnumpy(distances[k,]), 'o-')
        plt.title('Image Distance From Train Row %i to Other Train Rows' % k, size=16)
        plt.ylabel('Distance to Train Row %i' % k, size=14)
        plt.xlabel('Index Sorted by Distance to Train Row %i' % k, size=14)
        plt.show()

        cluster = train.loc[cupy.asnumpy(indices[k, :8])]
        displayDF(cluster, random=False, ROWS=2, COLS=4)


if __name__ == '__main__':
    restrict_tensorflow_mem_with_gpu_ram()
    train = load_train_data()
    # displayDF(train)
    # display_duplicated_items()
    cv2_view()