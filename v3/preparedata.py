import csv


codes = []


# 计算1-4月份特征保存在features.csv中
def get_features():
    holidays = [0, 1, 2, 41, 44, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 93, 94, 95]

    def get_date_in_month(day):
        if day <= 31:
            return day
        elif day <= 59:
            return day - 31
        elif day <= 90:
            return day - 59
        else:
            return day - 90

    with open('timeseries_customers.csv') as customers_file,\
     open('timeseries_discounts.csv') as discounts_file,\
     open('features.csv', 'w', newline='') as output_file:
        input_customers = csv.reader(customers_file)
        input_discounts = csv.reader(discounts_file)
        output_csv = csv.writer(output_file)
        next(input_customers)
        next(input_discounts)
        output_csv.writerow(['class', 'day_in_week', 'day_in_month', 'holiday', 'discount', 'label'])      # 中类特征
        for row in input_customers:
            class_code = row[0]
            discount_row = next(input_discounts)
            for day in range(1, 121):
                feature_row = []
                feature_row.append(class_code)
                day_in_week = day % 7 + 4
                feature_row.append(str(day_in_week))
                feature_row.append(str(get_date_in_month(day)))
                if day in holidays:
                    feature_row.append('1')
                else:
                    feature_row.append('0')
                feature_row.append(discount_row[day])
                feature_row.append(row[day])
                output_csv.writerow(feature_row)


def divide_train_test_set():
    with open('features.csv') as input_file,\
     open('train.csv', 'w', newline='') as train_file,\
     open('test.csv', 'w', newline='') as test_file:
        input_csv = csv.reader(input_file)
        train_csv = csv.writer(train_file)
        test_csv = csv.writer(test_file)
        next(input_csv)
        day = 0
        for row in input_csv:
            if day < 100:
                train_csv.writerow(row)
                day += 1
            else:
                test_csv.writerow(row)
                day = (day + 1) % 120


# 计算5月份特征并保存在May_input.csv中，其中大类最后一个特征（大类中中类的预测销量之和）需一边预测一边修改
def compute_May_features():
    def codes_list_out():
        global codes
        codes = [0]
        with open('commit_empty.csv') as native_set_file:
            native_csv = csv.reader(native_set_file)
            next(native_csv)
            for row in native_csv:
                if row[0] != codes[-1]:
                    codes.append(row[0])
        codes = codes[1:]
        print(codes)

    codes_list_out()
    with open('May_input.csv', 'w', newline='') as output_file:
        output_csv = csv.writer(output_file)
        for code in codes:
            for day in range(1, 31):
                feature = [code, str(day % 7 + 4), str(day), '0', '0']
                if len(code) == 2:      # 大类
                    feature.append('0')
                output_csv.writerow(feature)


if __name__ == '__main__':
    get_features()
    divide_train_test_set()
    compute_May_features()
