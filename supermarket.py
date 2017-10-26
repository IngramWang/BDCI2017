# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost
import numpy
import csv
import math

def getData(csvReader, trainCount, testCount):
    trainData = []
    testData = []
    trainLabel = []
    testLabel = []
    try:
        for x in range(0, trainCount):
            row = csvReader.next()
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8]), float(row[9]), float(row[10])]
            trainData.append(data)
            trainLabel.append(float(row[12]))
        for x in range(0, testCount):
            row = csvReader.next()
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8]), float(row[9]), float(row[10])]
            testData.append(data)
            testLabel.append(float(row[12]))
        return int(row[0]), trainData, trainLabel, testData, testLabel
    except StopIteration:
        return 0, [], [], [], []

f = open("data.csv", "r")
f_csv = csv.reader(f)

while (True):
    midclass, trD, trL, teD, teL = getData(f_csv, 100, 16)
    if (midclass == 0):
        break
    else:
        dtrain = xgboost.DMatrix(numpy.array(trD), label=numpy.array(trL))
        params = {"objective": "reg:linear"}
        gbm = xgboost.train(dtrain=dtrain, params=params)
        teP = gbm.predict(xgboost.DMatrix(numpy.array(teD)))
        bias = 0.0
        for i in range(0, 16):
            bias += (teP[i]-teL[i])*(teP[i]-teL[i]);
        bias = math.sqrt(bias/16)
        print "(Midclass %d predict finished, accuracy: %f)" % (midclass, bias)
        
f.close()