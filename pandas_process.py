import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2


file_result = './output/submission_0.31.csv'


def view_result_group():
    train = pd.read_csv('./shopee-product-matching/train.csv')
    image_name = train["image"].values
    submit_data = pd.read_csv(file_result)
    print(submit_data.head(10))
    list_id_image = submit_data.posting_id.values
    list_id_group = np.zeros(len(list_id_image))
    matches = submit_data["matches"].values

    start_time = time.time()
    for i in range(len(matches)):
        print("Processing...{}/{}".format(i, len(matches)))
        name_group = len(np.unique(list_id_group))
        data = matches[i].split(" ")
        if list_id_group[i] == 0.:
            list_id_group[i] = name_group
        for img_id in data:
            if list_id_image[i] == img_id:
                continue
            index_match = np.where(list_id_image == img_id)[0][0]
            if list_id_group[index_match] == 0.:
                list_id_group[index_match] = name_group
    print("process cost:", time.time() - start_time)
    submit_data.insert(submit_data.shape[-1], "image_name", image_name, True)
    submit_data.insert(submit_data.shape[-1], "group_name", list_id_group, True)
    print(submit_data.head(10))
    submit_data.to_csv(file_result[:-4] + "_group.csv", index=False)


def main():
    path = './shopee-product-matching/train_images/'
    train = pd.read_csv(file_result[:-4] + "_group.csv")
    print('train shape is', train.shape)
    train.head()
    groups = train.group_name.value_counts()
    print(groups.shape)
    size_image = 100
    num_col = 10
    for k in range(15):
        print('#' * 40)
        print('### TOP %i DUPLICATED ITEM:' % (k + 1), groups.index[k])

        top = train.loc[train.group_name == groups.index[k]]
        image_name_list = top.image_name.values

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
        print(combine_image.shape)
        print('#' * 40)
        #     mng = plt.get_current_fig_manager()
        #     mng.full_screen_toggle()
        plt.figure(figsize=(6 * 3.13, 4 * 3.13))
        plt.imshow(combine_image)
        plt.show()


def compare_target_output():
    target_data = pd.read_csv('./shopee-product-matching/train.csv')
    groups_output = target_data.label_group.value_counts()
    print(groups_output.shape)


    path = './shopee-product-matching/train_images/'
    output_data = pd.read_csv(file_result[:-4] + "_group.csv")
    print('train shape is', output_data.shape)
    output_data.head()
    groups_target = output_data.group_name.value_counts()
    print(groups_target.shape)
    size_image = 100
    num_col = 10
    for k in range(15):
        print('#' * 40)
        print('### TOP %i DUPLICATED ITEM:' % (k + 1), groups_target.index[k])

        top_output = output_data.loc[output_data.group_name == groups_target.index[k]]
        image_name_list_output = top_output.image_name.values

        # TOP 5 duplicated
        for k in range(15):
            print('#' * 40)
            print('### TOP %i DUPLICATED ITEM:' % (k + 1), groups_output.index[k])
            print('#' * 40)
            top_target = target_data.loc[target_data.label_group == groups_output.index[k]]
            image_name_list_target = top_target.image.values
            list_image_the_same1 = np.intersect1d(image_name_list_output, image_name_list_target)
            list_image_the_same2 = np.intersect1d(image_name_list_target, image_name_list_output)
            if len(list_image_the_same2) ==0 and len(list_image_the_same1) == 0:
                continue
            else:
                list_image_the_diff = np.setdiff1d(image_name_list_output, image_name_list_target)
                for image_name in list_image_the_diff:
                    arr_image = cv2.imread(path + image_name)
                    arr_image = cv2.cvtColor(arr_image, cv2.COLOR_BGR2RGB)
                    arr_image = cv2.resize(arr_image, (size_image, size_image), interpolation=cv2.INTER_AREA)
                    # plt.figure(figsize=(6 * 3.13, 4 * 3.13))
                    plt.imshow(arr_image)
                    plt.show()

                    a = 0




if __name__ == '__main__':
    # train = pd.read_csv('./shopee-product-matching/train.csv')
    # tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    # train['target'] = train.label_group.map(tmp)
    # print('train shape is', train.shape)
    # train.head()

    # view_result_group()
    # main()
    compare_target_output()