# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost
import numpy
import csv
import math

larclasPred = {}
larclasLabl = {}
totalBias = 0
totalCount = 0

def getData(csvReader, trainCount, testCount):
    trainData = []
    testData = []
    trainLabel = []
    testLabel = []
    try:
        for x in range(0, trainCount):
            row = csvReader.next()
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8])]
            trainData.append(data)
            trainLabel.append(float(row[11]))
        for x in range(0, testCount):
            row = csvReader.next()
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8])]
            testData.append(data)
            testLabel.append(float(row[11]))
        return int(row[0]), trainData, trainLabel, testData, testLabel
    except StopIteration:
        return 0, [], [], [], []
    
def dataLog(midclass, accuracy, trainLabl, testPred, testLabl):
    with open('compare.csv', 'ab') as f:
        writer = csv.writer(f)
        count = 1
        writer.writerow([midclass, accuracy])
        for x in trainLabl:
            writer.writerow([count, x])
            count += 1
        for x in range(0, len(testPred)):
            writer.writerow([count, testLabl[x], testPred[x]])
            count += 1

def test():
    global larclasPred, larclasLabl, totalBias, totalCount   
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
            totalBias += bias
            totalCount += 16
            bias = math.sqrt(bias/16)
            print "(Midclass %d predict finished, accuracy: %f)" % (midclass, bias)
            larclass = int(midclass/100)
            if (larclass in larclasPred):
                for i in range(0, 16):
                    larclasPred[larclass][i] += teP[i]
                    larclasLabl[larclass][i] += teL[i]
            else:
                larclasPred[larclass] = teP
                larclasLabl[larclass] = teL
            dataLog(midclass, bias, trL, teP, teL)                
    for larclass in larclasPred:
        bias = 0.0
        for i in range(0, 16):
            d = larclasLabl[larclass][i] - larclasPred[larclass][i]
            bias += d*d;
        totalBias += bias
        totalCount += 16
        bias = math.sqrt(bias/16)
        print "(Larclass %d predict finished, accuracy: %f)" % (larclass, bias)  
        
    totalBias = math.sqrt(totalBias/totalCount)
    print "(Predict finished, accuracy: %f)" % (totalBias)        
    f.close()
    
def submit(): 
    global larclasPred
    f1 = open("data.csv", "r")
    data_csv = csv.reader(f1)
    f2 = open("submit.csv", "r")
    submit_csv = csv.reader(f2)
    submit_csv.next()
    
    goal = []
    for i in range(1, 31):
        x = [i, (i+4)%7, 0, 0, 0, 0]
        if (x[1] == 6 or x[1]==0):
            x[3] = 1
        elif (x[1] == 5):
            x[2] = 1
        goal.append(x)
    goal[0][3] = 1
    goal[0][2] = 0
    
    while (True):
        midclass, trD, trL, teD, teL = getData(data_csv, 116, 0)
        if (midclass == 0):
            break
        else:
            dtrain = xgboost.DMatrix(numpy.array(trD), label=numpy.array(trL))
            params = {"objective": "reg:linear"}
            gbm = xgboost.train(dtrain=dtrain, params=params)
            teP = gbm.predict(xgboost.DMatrix(numpy.array(goal)))
            
            for x in teP:
                row = submit_csv.next()
                if (int(row[0]) != midclass):
                    raise KeyError
                with open('submit1.csv', 'ab') as f:
                    writer = csv.writer(f)
                    writer.writerow([row[0], row[1], x])
            
            larclass = int(midclass/100)
            if (larclass in larclasPred):
                for i in range(0, 30):
                    larclasPred[larclass][i] += teP[i]
            else:
                larclasPred[larclass] = teP  
    oldLC = 0            
    for row in submit_csv:
        larclass = int(row[0])
        if larclass != oldLC:
            oldLC = larclass
            i = 0
        with open('submit1.csv', 'ab') as f:
            writer = csv.writer(f)
            writer.writerow([row[0], row[1], larclasPred[larclass][i]]) 
        i+=1
    f1.close()
    f2.close()
            
test()