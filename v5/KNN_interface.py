# version 5 KNN算法的调用接口

from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import csv


# data: array of float, 销量数据
# pred_length: int，需预测的天数
# D_window（窗口长度）和max_k（最大的k值）
def knn(data, pred_length, D_window=14, max_k=7):
    if pred_length + D_window >= len(data):
        print('ERROR: pred_length or D_window too long')
        return None

    ret_ypred = []
    for h in range(4):
        train_feature, train_label = get_train_set(data, h, D_window, pred_length)

        e_LOO_arr = np.zeros(max_k)
        for k in range(2, max_k + 1):
            model = KNeighborsRegressor(n_neighbors=k, weights='uniform', algorithm='auto')
            model.fit(train_feature, train_label)

            # 获取k近邻
            dist_list, index_list = model.kneighbors([data[0 - D_window:]])
            k_neighbor_label = []
            for i in index_list[0]:
                k_neighbor_label.append(train_label[i])

            # 基于k近邻的预测值
            ypred = model.predict([data[0-D_window:]])
            ypred = np.asarray(list(map(round, ypred[0])))

            # 计算e_LOO
            e_LOO_arr[k-1] = LOO(k_neighbor_label, ypred, k)

        # 取e_LOO最小的k值
        k_min = np.argmin(e_LOO_arr[1:]) + 2
        model = KNeighborsRegressor(n_neighbors=k_min, weights='uniform', algorithm='auto')
        model.fit(train_feature, train_label)
        ypred = model.predict([data[0 - D_window:]])
        ret_ypred += list(map(round, ypred[0]))

    return np.asarray(ret_ypred)


def get_train_set(train_data, h, D, pred_length):
    feature, label = [], []
    block_len = int(pred_length / 4)
    if h != 3:
        for i in range(len(train_data) - D - block_len * (h + 1) + 1):
            feature.append(train_data[i:i + D])
            label.append(train_data[i + D + block_len * h:i + D + block_len * h + block_len])
    else:
        for i in range(len(train_data) - D - pred_length + 1):
            feature.append(train_data[i:i + D])
            label.append(train_data[i + D + 3 * block_len:i + D + pred_length])
    return np.array(feature), np.array(label)


# 计算LOO，用于k（近邻数）的选择
def LOO(k_neighbor_label, ypred, k):
    ret = 0
    for neighbor in k_neighbor_label:
        ret = ret + ((neighbor - ypred) ** 2).sum()
    ret = ret * k / (k - 1)**2
    # ret = ret / (k)**2
    return ret


def test():
    with open('timeseries_customers_processed.csv') as input_file:
        input_csv = csv.reader(input_file)
        next(input_csv)
        row = next(input_csv)
        data = list(map(float, row[1:]))
        print(knn(data, 30))


if __name__ == '__main__':
    test()
