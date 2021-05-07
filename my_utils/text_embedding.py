# Import packages
import pandas as pd
import gc
import numpy as np
import time
import cudf, cuml, cupy
from cuml.feature_extraction.text import TfidfVectorizer
print('RAPIDS', cuml.__version__)


def read_dataset():
    cu_df = cudf.read_csv("../shopee-product-matching/train.csv")
    test_images_path = "../shopee-product-matching/test_images/" + cu_df["image"]
    return cu_df, test_images_path


# Get text embedding from cuml
def get_text_embeddings(cu_df):
    model = TfidfVectorizer(stop_words='english', binary=True)
    text_embeddings = model.fit_transform(cu_df.title).toarray()
    del model
    return text_embeddings


def method(lists, value):
    return next((i for i,v in enumerate(lists) if value in v), -1)


def find_similar_titles(text_embeddings):
    test = pd.read_csv('../shopee-product-matching/train.csv')
    print(test.head())
    preds = []
    index_roup = []
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
            #         print("cupy.where(cts[k,]>0.7)", cupy.where(cts[k,]>0.7)[0])
            IDX = cupy.where(cts[k, ] > 0.7)[0]
            #         print("IDX", IDX)
            #         print("cupy.asnumpy(IDX)", test.iloc[cupy.asnumpy(IDX)].posting_id)
            o = test.iloc[cupy.asnumpy(IDX)].posting_id.values
            flask = False
            for i in range(len(o)):
                if o[i] in list(pd.core.common.flatten(preds)):
                    index = method(preds, o[i])
                    if index != -1:
                        index_roup.append(index)
                        flask = True
                        break

            if flask is False:
                index_roup.append(len(preds))
            preds.append(o)
    del text_embeddings
    _ = gc.collect()

    test['preds'] = preds
    test.head()

    return test


def combine_for_sub(row):
    x = np.concatenate([row.preds])
    return ' '.join(np.unique(x))


def write_csv_file(test):
    test[['posting_id', 'matches']].to_csv('submission.csv', index=False)
    sub = pd.read_csv('submission.csv')
    sub.head()


def main():
    # Main function
    cu_df, _ = read_dataset()
    start_time = time.time()
    text_embeddings = get_text_embeddings(cu_df)
    print("Get text embedding GPU cost", time.time() - start_time)
    print(text_embeddings.shape)
    print(text_embeddings[0])
    test = find_similar_titles(text_embeddings)
    test['matches'] = test.apply(combine_for_sub, axis=1)
    write_csv_file(test)


if __name__ == '__main__':
    main()