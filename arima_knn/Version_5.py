import csv
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

from modify_submit import change_pred


def main_fun():
    class_codes = ['1201', '2011', '12', '15', '20', '22', '23', '30']
    with open('timeseries_customers_processed.csv') as input_file:
        input_csv = csv.reader(input_file)
        next(input_csv)
        for row in input_csv:
            if row[0] in class_codes:
                # MIMO_KNN_test(row)
                # MIMO_KNN_LOO_test(row)
                MIMO_KNN_LOO_May(row)


# 划分数据集测试不同参数（D_window, k），没有预测5月份销量
def MIMO_KNN_test(data):
    code = data[0]
    data = list(map(float, data[1:]))
    train_data = data[:90]
    test_data = data[90:]

    # 对4个时间段分别训练模型，时间段分别为7天、7天、7天、9天
    D_window = 14
    for h in range(4):
        train_feature, train_label = get_train_set(train_data, h, D_window)
        y_label = get_test_label(test_data, h)

        for k in range(1, 8):
            model = KNeighborsRegressor(n_neighbors=k, weights='uniform', algorithm='auto')
            model.fit(train_feature, train_label)

            ypred = model.predict([train_data[0-D_window:]])
            ypred = np.array(list(map(round, ypred[0])))

            rmse = np.sqrt(((ypred - y_label) ** 2).mean())
            print(code, '  h=', h, '  k=', k, '  rmse=', rmse)


# 划分数据集，实现论文里的方法，没有预测5月份销量
def MIMO_KNN_LOO_test(data):
    code = data[0]
    data = list(map(float, data[1:]))
    train_data = data[:90]
    test_data = data[90:]

    # 对4个时间段分别训练模型，时间段分别为7天、7天、7天、9天
    D_window = 14
    max_k = 7
    for h in range(4):
        train_feature, train_label = get_train_set(train_data, h, D_window)
        y_label = get_test_label(test_data, h)

        e_LOO_arr = np.zeros(max_k)
        for k in range(2, max_k + 1):
            model = KNeighborsRegressor(n_neighbors=k, weights='uniform', algorithm='auto')
            model.fit(train_feature, train_label)

            # 获取k近邻
            dist_list, index_list = model.kneighbors([train_data[0 - D_window:]])
            k_neighbor_label = []
            for i in index_list[0]:
                k_neighbor_label.append(train_label[i])

            # 基于k近邻的预测值
            ypred = model.predict([train_data[0-D_window:]])
            ypred = np.asarray(list(map(round, ypred[0])))
            rmse = np.sqrt(((ypred - y_label) ** 2).mean())
            print(code, '  h=', h, '  k=', k, '  rmse=', rmse)

            # 计算e_LOO
            e_LOO_arr[k-1] = LOO(k_neighbor_label, ypred, k)

        # 取e_LOO最小的k值
        k_min = np.argmin(e_LOO_arr[1:]) + 2
        print('k_min=', k_min)


# 使用整个数据集，实现论文里的方法，预测5月份销量
def MIMO_KNN_LOO_May(data):
    code = data[0]
    data = list(map(float, data[1:]))

    D_window = 14
    max_k = 7
    pred_May = []
    for h in range(4):
        train_feature, train_label = get_train_set(data, h, D_window)
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
            ypred = model.predict([data[0 - D_window:]])
            ypred = np.asarray(list(map(round, ypred[0])))

            # 计算e_LOO
            e_LOO_arr[k - 1] = LOO(k_neighbor_label, ypred, k)

        # 取e_LOO最小的k值
        k_min = np.argmin(e_LOO_arr[1:]) + 2

        # 令k=k_min，做预测
        model = KNeighborsRegressor(n_neighbors=k_min, weights='uniform', algorithm='auto')
        model.fit(train_feature, train_label)
        ypred = model.predict([data[0 - D_window:]])
        ypred = list(map(round, ypred[0]))
        pred_May = pred_May + ypred

    print(pred_May)
    # 替换文件里编码为code的预测值
    change_pred(code, pred_May)


# 计算LOO，用于k（近邻数）的选择
def LOO(k_neighbor_label, ypred, k):
    ret = 0
    for neighbor in k_neighbor_label:
        ret = ret + ((neighbor - ypred) ** 2).sum()
    ret = ret * k / (k - 1)**2
    # ret = ret / (k)**2
    return ret


def get_train_set(train_data, h, D):
    feature, label = [], []
    if h != 3:
        for i in range(len(train_data) - D - 7 * (h+1) + 1):
            feature.append(train_data[i:i+D])
            label.append(train_data[i+D+7*h:i+D+7*h+7])
    else:
        for i in range(len(train_data) - D - 30 + 1):
            feature.append(train_data[i:i+D])
            label.append(train_data[i+D+21:i+D+30])
    return np.array(feature), np.array(label)


def get_test_label(test_data, h):
    if h != 3:
        return test_data[7*h:7*h+7]
    else:
        return test_data[21:]


if __name__ == '__main__':
    main_fun()
