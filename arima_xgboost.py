# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost as xgb
import arimaPredicter
import dataLoader

from numpy import array
from numpy import zeros
import csv
import math
import datetime as dt

larclasPred = {}
larclasLabl = {}
totalBias = 0
totalCount = 0

dtIndex = [dt.datetime(2015,1,x) for x in range(1, 32)]
dtIndex = dtIndex + [dt.datetime(2015,2,x) for x in (range(1, 29))]
dtIndex = dtIndex + [dt.datetime(2015,3,x) for x in range(1, 32)]
dtIndex = dtIndex + [dt.datetime(2015,4,x) for x in (range(1, 31))]

modelChoose = []
lcModelChoose = []
arimaParaChoose = {}

ap = arimaPredicter.predicter()
ap.setIndex(dtIndex)
    
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

def xgboostPredict(trainData, trainLabel, dataToPredict):
    dtrain = xgb.DMatrix(trainData, trainLabel)
    params = {"objective": "reg:linear"}
    gbm = xgb.train(dtrain=dtrain, params=params)
    return gbm.predict(xgb.DMatrix(dataToPredict))

def modelselect(trainSize, testSize, skipSize = 0):
    global larclasPred, totalBias, totalCount, modelChoose, lcModelChoose 
    larclasPred = {}
    totalBias = 0
    totalCount = 0
    modelChoose = []
    lcModelChoose = []
    
    loader = dataLoader.loader("datam.csv", "lcdatam.csv")
    loader.setSize(trainSize, testSize, skipSize)
        
    # middle class
    teD = []
    for i in range(31-testSize, 31):
        x = [i, (i+2)%7, 0, 0, 0, 0]
        if (x[1] == 6 or x[1]==0):
            x[3] = 1
        elif (x[1] == 5):
            x[2] = 1
        teD.append(x)
    while (True):
        midclass, trD, trL, _, teL = loader.getNextMidClass() 
        if (midclass == 0):
            break
        else:

            # sarima model
            try:
                model = ap.sarimaTrain(midclass, trL, teL)
                teP1 = ap.sarimaPredict(model, testSize)
            except:
                teP1 = zeros(testSize)
            
            # xgboost model
            try:
                teP2 = xgboostPredict(array(trD), array(trL), array(teD))
            except:
                teP2 = zeros(testSize)
            
            # just zero
            teP3 = zeros(testSize)

            # count bias of midclass and update larclass
            label = array(teL)
            larclass = int(midclass/100)
            totalCount += testSize
  
            bias1 = sum((teP1-label)*(teP1-label))
            bias2 = sum((teP2-label)*(teP2-label))
            bias3 = sum((teP3-label)*(teP3-label))
            if (bias3 <= bias1 and bias3 <= bias2):
                totalBias += bias3
                bias3 = math.sqrt(bias3/testSize)
                print "(Midclass %d select ZERO, accuracy: %f)" % (midclass, bias3)
                modelChoose.append(3)
                if (larclass in larclasPred):
                    larclasPred[larclass] += teP3
                else:
                    larclasPred[larclass] = teP3
            elif (bias1 <= bias2):
                totalBias += bias1
                bias1 = math.sqrt(bias1/testSize)
                print "(Midclass %d select SARIMA, accuracy: %f)" % (midclass, bias1)
                modelChoose.append(1)
                if (larclass in larclasPred):
                    larclasPred[larclass] += teP1
                else:
                    larclasPred[larclass] = teP1
            else:
                totalBias += bias2
                bias2 = math.sqrt(bias2/testSize)
                print "(Midclass %d select XGBOOST, accuracy: %f)" % (midclass, bias2)
                modelChoose.append(2)
                if (larclass in larclasPred):
                    larclasPred[larclass] += teP2
                else:
                    larclasPred[larclass] = teP2
                    
    # large class
    teD = []
    for i in range(31-testSize, 31):
        x = [i, (i+2)%7, 0, 0, 0]
        if (x[1] == 6 or x[1]==0):
            x[3] = 1
        elif (x[1] == 5):
            x[2] = 1
        teD.append(x)
    while (True):
        larclass, trD, trL, _, teL = loader.getNextLarClass()  
        if (larclass == 0):
            break
        else:

            # sarima model
            try:
                model = ap.sarimaTrain(larclass, trL, teL)
                teP1 = ap.sarimaPredict(model, testSize)
            except:
                teP1 = zeros(testSize)
            
            # xgboost model
            try:
                teP2 = xgboostPredict(array(trD), array(trL), array(teD))
            except:
                teP2 = zeros(testSize)
            
            # sum of midclasses
            teP3 = larclasPred[larclass]

            # count bias of midclass and update larclass
            label = array(teL)
            totalCount += testSize
  
            bias1 = sum((teP1-label)*(teP1-label))
            bias2 = sum((teP2-label)*(teP2-label))
            bias3 = sum((teP3-label)*(teP3-label))
            if (bias3 <= bias1 and bias3 <= bias2):
                totalBias += bias3
                bias3 = math.sqrt(bias3/testSize)
                print "(Larclass %d select SUM, accuracy: %f)" % (larclass, bias3)
                lcModelChoose.append(3)
            elif (bias1 <= bias2):
                totalBias += bias1
                bias1 = math.sqrt(bias1/testSize)
                print "(Larclass %d select SARIMA, accuracy: %f)" % (larclass, bias1)
                lcModelChoose.append(1)
            else:
                totalBias += bias2
                bias2 = math.sqrt(bias2/testSize)
                print "(Larclass %d select XGBOOST, accuracy: %f)" % (larclass, bias2)
                lcModelChoose.append(2)

    totalBias = math.sqrt(totalBias/totalCount)
    print "(Predict finished, accuracy: %f)" % (totalBias)        
    loader.closeFiles()
    
def submit(trainSize): 
    global larclasPred
    larclasPred = {}
    f1 = open("submit.csv", "r")
    submit_csv = csv.reader(f1)
    submit_csv.next()
    f2 = open('submit1.csv', 'wb')
    writer = csv.writer(f2)
    
    loader = dataLoader.loader("datam.csv", "lcdatam.csv")
    loader.setSize(trainSize)
    
    # middle class
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
    
    current = 0
    
    while (True):
        midclass, trD, trL, teD, teL = loader.getNextMidClass()
        if (midclass == 0):
            break
        else:
            if (modelChoose[current] == 1):
                try:
                    model = ap.sarimaTrain(midclass, trL)
                    teP = ap.sarimaPredict(model, 30)
                except:
                    print("%d: failed to use arima, use xgboost instead" % midclass)
                    teP = xgboostPredict(array(trD), array(trL), array(goal))
            elif (modelChoose[current] == 2):
                teP = xgboostPredict(array(trD), array(trL), array(goal))
            else:
                teP = zeros(30)
            current += 1
            
            for x in teP:
                x_int = round(x)
                row = submit_csv.next()
                if (int(row[0]) != midclass):
                    raise KeyError
                writer.writerow([row[0], row[1], x_int])
            
            # count larclass
            larclass = int(midclass/100)
            if (larclass in larclasPred):
                larclasPred[larclass] += teP
            else:
                larclasPred[larclass] = teP  
    
    # large class
    goal = []
    for i in range(1, 31):
        x = [i, (i+4)%7, 0, 0, 0]
        if (x[1] == 6 or x[1]==0):
            x[3] = 1
        elif (x[1] == 5):
            x[2] = 1
        goal.append(x)
    goal[0][3] = 1
    goal[0][2] = 0
    
    current = 0
    
    while (True):
        larclass, trD, trL, teD, teL = loader.getNextLarClass()
        if (larclass == 0):
            break
        else:
            if (lcModelChoose[current] == 1):
                try:
                    model = ap.sarimaTrain(larclass, trL)
                    teP = ap.sarimaPredict(model, 30)
                except:
                    print("%d: failed to use arima, use xgboost instead" % larclass)
                    teP = xgboostPredict(array(trD), array(trL), array(goal))
            elif (lcModelChoose[current] == 2):
                teP = xgboostPredict(array(trD), array(trL), array(goal))
            else:
                teP = larclasPred[larclass]
            current += 1

            # write file - midclass
            for x in teP:
                x_int = round(x)
                row = submit_csv.next()
                if (int(row[0]) != larclass):
                    raise KeyError
                writer.writerow([row[0], row[1], x_int])

    f1.close()
    f2.close()
    loader.closeFiles()
           
modelselect(75, 14, 31)
"""
with open("report.txt", "w") as f:
    for clas in arimaParaChoose:
        f.writelines("class %d: (%d,%d)\n" % (clas, arimaParaChoose[clas][0], arimaParaChoose[clas][1]))
"""
submit(120)