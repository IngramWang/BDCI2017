# 把不同模型结果合并在一个文件中

import csv

commit_codes = []


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


def get_day(date):
    date = int(date)
    if date < 20150132:
        return date - 20150100
    elif date < 20150229:
        return date - 20150200 + 31
    elif date < 20150332:
        return date - 20150300 + 59
    else:
        return date - 20150400 + 90


def merge_file():
    features = {}
    with open('five_fold_feature_v3.csv') as input_file:
        input_csv = csv.reader(input_file)
        next(input_csv)
        for row in input_csv:
            if row[0] not in features:
                features[row[0]] = [row]
            else:
                features[row[0]].append(row)
    with open('five_fold_feature_xgboost.csv') as input_file:
        input_csv = csv.reader(input_file)
        for row in input_csv:
            if row[0] in features:
                features[row[0]][get_day(row[1])-1] = features[row[0]][get_day(row[1])-1] + row[2:]
    # 最后一列是label
    with open('timeseries_customers.csv') as input_file:
        input_csv = csv.reader(input_file)
        for row in input_csv:
            if row[0] in features:
                for day in range(120):
                    features[row[0]][day].append(row[day+1])
    with open('merged_feature.csv', 'w', newline='') as output_file:
        output_csv = csv.writer(output_file)
        for code in commit_codes:
            for row in features[code]:
                output_csv.writerow(row)


def divide_train_test_set():
    with open('merged_feature.csv') as input_file,\
     open('train.csv', 'w', newline='') as train_file,\
     open('test.csv', 'w', newline='') as test_file:
        input_csv = csv.reader(input_file)
        train_csv = csv.writer(train_file)
        test_csv = csv.writer(test_file)
        day = 0
        for row in input_csv:
            if day < 100:
                train_csv.writerow(row)
                day += 1
            else:
                test_csv.writerow(row)
                day = (day + 1) % 120


if __name__ == '__main__':
    codes_list_out()
    merge_file()
    divide_train_test_set()
