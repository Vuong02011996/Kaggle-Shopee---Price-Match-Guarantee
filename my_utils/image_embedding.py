import pandas as pd
import numpy as np


def test():
    train = pd.read_csv('../shopee-product-matching/train.csv')
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    label_group_arr = train.label_group.to_numpy()
    list_label_group_sort = []
    for k in sorted(tmp, key=lambda k: len(tmp[k]), reverse=True):
        list_label_group_sort.append(k)
    print(list_label_group_sort[0])
    for label_group in list_label_group_sort:
        index_group = np.where(label_group_arr == label_group)[0]

    for i in range(10):
        print(len(tmp[list_label_group_sort[i]]))
    train['target'] = train.label_group.map(tmp)
    print('train shape is', train.shape )
    train.head()
    a = 0


def show_chris_dtteo():
    train = pd.read_csv('../shopee-product-matching/train.csv')
    print('train shape is', train.shape)
    train.head()
    groups = train.label_group.value_counts()
    print(groups.shape)


def main():
    submit = pd.read_csv('../my_utils/submission.csv')
    print(submit.head(10))
    matches = submit["matches"].values
    for i in range(len(matches)):
        data = matches[i].split(" ")
        if len(data) > 1:
            for j in range(len(matches[i+1:])):
                next_data = matches[j].split(" ")
                if len(next_data) > 1:
                    pass

        else:
            matches[i] += " grouped_" + str(i + 1000)
    a = 0


if __name__ == '__main__':
    # test()
    show_chris_dtteo()
