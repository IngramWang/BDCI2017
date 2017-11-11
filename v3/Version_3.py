from sklearn.ensemble import RandomForestRegressor
import numpy as np
import csv


mid_class_num = 134
large_class_num = 14
class_codes = []
train_set_x = {}
train_set_y = {}
test_set_x = {}
test_set_y = {}
May_set_x = {}
May_set_y = {}
large_codes = ['10', '11', '12', '13', '15', '20', '21', '22', '23', '30', '31', '32', '33', '34']
commit_codes = []

accumulate_err = 0


# 载入训练和测试模型的数据（不包括5月份的）
def load_data():
    with open('train.csv') as input_file:
        input_csv = csv.reader(input_file)
        day = 0
        for row in input_csv:
            code = row[0]
            if day == 0:
                class_codes.append(code)
                train_set_x[code] = []
                train_set_y[code] = []
            x = list(map(float, row[1:-1]))
            # 将大类的feature增加一项：预测的当天的对应中类customer之和， 初始化为0
            if code in large_codes:
                x.append(0)
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
            x = list(map(float, row[1:-1]))
            # 将大类的feature增加一项：预测的当天的对应中类customer之和， 初始化为0
            if code in large_codes:
                x.append(0)
            test_set_x[code].append(x)
            test_set_y[code].append(float(row[-1]))
            day = (day + 1) % 20


def load_May_data():
    with open('May_input.csv') as input_file:
        input_csv = csv.reader(input_file)
        day = 0
        for row in input_csv:
            code = row[0]
            if code in commit_codes:
                if day == 0:
                    May_set_x[code] = []
                May_set_x[code].append(list(map(float, row[1:])))
                day = (day + 1) % 30


# 修改大类feature的最后一项（大类中中类的预测销量和）
def modify_large_feature(type, class_code, pred):
    class_code = class_code[:2]
    if type == 'train':
        for day in range(len(train_set_x[class_code])):
            train_set_x[class_code][day][-1] += pred[day]
    if type == 'test':
        for day in range(len(test_set_x[class_code])):
            test_set_x[class_code][day][-1] += pred[day]


def train_test_eval(train_x, train_y, test_x, test_y, params=None):
    # train
    if params is None:
        rf = RandomForestRegressor()
    else:
        rf = RandomForestRegressor(n_estimators=params['n_estimators'], oob_score=params['oob_score'])
    rf.fit(train_x, train_y)

    # test
    ypred = np.asarray(list(map(round, rf.predict(test_x))))

    # evaluation
    rmse = np.sqrt(((test_y - ypred) ** 2).mean())
    global accumulate_err
    accumulate_err += np.sum((test_y - ypred) ** 2)

    # this is used for modifying large class feature
    train_predict = rf.predict(train_x)

    return rf, ypred, rmse, train_predict


# 为每一个类训练一个模型，如果params为None，则预测5月份的销量；否则用params测试，不预测5月份，并将结果RMSE写到 调参.txt 中
def run_for_classes(params=None):
    output = []
    for code in class_codes:
        if code not in commit_codes:
            continue
        model, ypred, rmse, train_predict = train_test_eval(train_set_x[code], train_set_y[code], test_set_x[code], test_set_y[code], params)
        if code in large_codes:
            modify_large_feature('train', code, train_predict)
            modify_large_feature('test', code, ypred)
        if params is None:
            print('class: ', code, '    RMSE: ', rmse)

            # prediction for May
            predict_May(model, code)

        else:
            output.append('class: ' + code + '  RMSE: ' + str(rmse) + '\n')

    if params is not None:
        global accumulate_err
        with open('调参.txt', 'a') as output_file:
            output_file.write('n_estimators=' + str(params['n_estimators']) + '  oob_score=' + str(params['oob_score']) + '\n')
            output_file.writelines(output)
            output_file.write('total RMSE: ' + str(accumulate_err / 2960))
        accumulate_err = 0


# 调参
def run_for_classes_params():
    for n_estimators in range(50, 160, 10):
        params = {'n_estimators': n_estimators, 'oob_score': False}
        run_for_classes(params)
        params = {'n_estimators': n_estimators, 'oob_score': True}
        run_for_classes(params)


def predict_May(rfmodel, code):
    ypred = rfmodel.predict(May_set_x[code])
    ypred = list(map(round, ypred))
    May_set_y[code] = ypred
    large_code = code[:2]
    for day in range(30):
        May_set_x[large_code][day][-1] += ypred[day]


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

if __name__ == '__main__':
    load_data()
    codes_list_out()
    load_May_data()

    # 不调参，用默认参数预测5月份，结果保存在字典May_set_y中
    run_for_classes()
    # write the predicted results of May
    with open('submit.csv', 'w', newline='') as output_file:
        output_csv = csv.writer(output_file)
        output_csv.writerow(['编码', '日期', '销量'])
        for code in commit_codes:
            for day in range(30):
                output_csv.writerow([code, str(20150501 + day), str(int(May_set_y[code][day]))])

    '''
    # 调参时调用
    run_for_classes_params()'''
