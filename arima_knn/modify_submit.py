import csv


# 读取原预测文件，预测结果取整再写回去
def get_round():
    rows = []
    with open('submit.csv') as input_file:
        input_csv = csv.reader(input_file)
        rows.append(next(input_csv))
        for row in input_csv:
            row[2] = str(int(round(float(row[2]))))
            rows.append(row)
    with open('submit.csv', 'w', newline='') as output_file:
        output_csv = csv.writer(output_file)
        for row in rows:
            output_csv.writerow(row)


# 将预测文件中编码为code的类别预测值用pred替换
def change_pred(code, pred):
    rows = []
    file_name = 'submit_WJ_2.csv'
    with open(file_name) as input_file:
        input_csv = csv.reader(input_file)
        rows.append(next(input_csv))
        i = 0
        for row in input_csv:
            if row[0] == code:
                rows.append([code, row[1], str(pred[i])])
                i += 1
            else:
                rows.append(row)
    with open(file_name, 'w', newline='') as output_file:
        output_csv = csv.writer(output_file)
        for row in rows:
            output_csv.writerow(row)


if __name__ == '__main__':
    get_round()
