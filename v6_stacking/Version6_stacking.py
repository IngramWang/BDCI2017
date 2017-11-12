import csv
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np


train_set_x = {}
train_set_y = {}
test_set_x = {}
test_set_y = {}
commit_codes = []


# 载入训练和测试模型的数据（不包括5月份的）
def load_data():
    with open('train.csv') as input_file:
        input_csv = csv.reader(input_file)
        day = 0
        for row in input_csv:
            code = row[0]
            if day == 0:
                train_set_x[code] = []
                train_set_y[code] = []
            x = list(map(float, row[2:-1]))
            train_set_x[code].append(x)
            train_set_y[code].append(float(row[-1]))
            day = (day + 1) % 100
    with open('test.csv') as input_file:
        input_csv = csv.reader(input_file)
        day = 0
        for row in input_csv:
            code = row[0]
            if day == 0:
                test_set_x[code] = []
                test_set_y[code] = []
            x = list(map(float, row[2:-1]))
            test_set_x[code].append(x)
            test_set_y[code].append(float(row[-1]))
            day = (day + 1) % 20


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


def train_test_eval():
    for code in commit_codes:
        # model = linear_model.LinearRegression()
        model = RandomForestRegressor()
        model.fit(train_set_x[code], train_set_y[code])
        ypred = model.predict(test_set_x[code])
        ypred = np.array(list(map(round, ypred)))
        rmse = np.sqrt(((test_set_y[code] - ypred) ** 2).mean())
        print(code, '  rmse=', rmse)


if __name__ == '__main__':
    codes_list_out()
    load_data()
    train_test_eval()
