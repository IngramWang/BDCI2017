# 五折交叉，用于v5的stacking

import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor


commit_codes = []
all_x = {}
all_y = {}
all_pred = {}


def load_all_data():
    global all_x, all_y
    with open('features.csv') as input_file:
        input_csv = csv.reader(input_file)
        next(input_csv)
        for row in input_csv:
            feature = list(map(float, row[1:-1]))
            if len(row[0]) == 2:
                feature.append(0)
            if row[0] not in all_x:
                all_x[row[0]] = [feature]
                all_y[row[0]] = [float(row[-1])]
            else:
                all_x[row[0]].append(feature)
                all_y[row[0]].append(float(row[-1]))


def get_day(date):
    date = int(date)
    if date < 20150132:
        return date - 20150100
    elif date < 20150229:
        return date - 20150200 + 31
    elif date < 20150332:
        return date - 20150300 + 59
    else:
        return date - 20150400 + 89


# 用不同参数调用five_fold_params_pred
def five_fold_pred():
    global all_pred
    for n_estimators in range(50, 160, 10):
        print('n_estimators= ', n_estimators)
        params = {'n_estimators': n_estimators, 'oob_score': False}
        five_fold_params_pred(params)
        # 写回文件
        output = []
        with open('five_fold_feature.csv') as input_file:
            input_csv = csv.reader(input_file)
            output.append(next(input_csv))
            for row in input_csv:
                output.append(row + [str(all_pred[row[0]][get_day(row[1])-1])])
        with open('five_fold_feature_v3.csv', 'w', newline='') as output_file:
            output_csv = csv.writer(output_file)
            for row in output:
                output_csv.writerow(row)

        # 清空all_pred
        all_pred = {}


# 用指定参数，5折交叉
def five_fold_params_pred(params):
    global commit_codes, all_pred
    for code in commit_codes:
        if code not in all_pred:
            all_pred[code] = np.zeros(120)
        if code not in all_x:              # 部分商品类原始数据里没有
            continue
        for i in range(5):
            train_x, train_y, test_x = get_fold_set(code, i)
            rf = RandomForestRegressor(n_estimators=params['n_estimators'], oob_score=params['oob_score'])
            rf.fit(train_x, train_y)
            ypred = rf.predict(test_x)
            # 存入all_pred
            for index in range(24):
                all_pred[code][i*24+index] = ypred[index]

        # 修改对应大类的最后一个特征值
        large_code = code[:2]
        for day in range(120):
            all_x[large_code][day][-1] += all_pred[code][day]


def get_fold_set(code, fold_index):
    train_x, train_y, test_x = [], [], []
    for i in range(120):
        if (i >= fold_index * 24) and (i < (fold_index + 1) * 24):
            test_x.append(all_x[code][i])
        else:
            train_x.append(all_x[code][i])
            train_y.append(all_y[code][i])
    return train_x, train_y, test_x


# 获取提交文件中需要提交的codes，保存在commit_codes中
def codes_list_out():
    global commit_codes
    codes = [0]
    with open('commit_empty.csv') as native_set_file:
        native_csv = csv.reader(native_set_file)
        next(native_csv)
        for row in native_csv:
            if row[0] != codes[-1]:
                codes.append(row[0])
    commit_codes = codes[1:]


# 初始化结果文件
def initialize_file():
    global commit_codes
    with open('five_fold_feature.csv', 'w', newline='') as output_file:
        output_csv = csv.writer(output_file)
        output_csv.writerow(['code', 'date', 'models'])
        for code in commit_codes:
            for date in range(20150101, 20150132):
                output_csv.writerow([code, str(date)])
            for date in range(20150201, 20150229):
                output_csv.writerow([code, str(date)])
            for date in range(20150301, 20150332):
                output_csv.writerow([code, str(date)])
            for date in range(20150401, 20150431):
                output_csv.writerow([code, str(date)])


if __name__ == '__main__':
    codes_list_out()
    initialize_file()
    load_all_data()
    five_fold_pred()
