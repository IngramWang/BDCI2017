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

ap = arimaPredicter.predicter()
ap.createIndex(dt.datetime(2015,1,1), 243)

modelChoose = {}

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

def xgboostPredict(trainData, trainLabel, dataToPredict, 
                   params = {"objective":"reg:linear", "max_depth":1, "gamma":2}):
    dtrain = xgb.DMatrix(trainData, trainLabel)
    gbm = xgb.train(dtrain=dtrain, params=params)
    return gbm.predict(xgb.DMatrix(dataToPredict))

def simulateFeature(trainData, musk):
    for feature in trainData:
        for i in musk:
            feature[i] = 0
            
def createFeature(date_from, length, zeros, DictHoilday, DictBeforeHoilday, 
                DictWorkday):
    delta = dt.timedelta(days=1)
    now = date_from
    index = []
    for i in range(0, length):
        index.append(now)
        now = now + delta
    feature = []
    empty = [0 for x in range(0, zeros+4)]
    for i in range(0, length):
        x = empty[:]
        x[0] = index[i].day
        x[1] = (index[i].weekday() + 1) % 7
        dayCount = i + 1
        if (dayCount in DictHoilday):
            x[3] = 1
        elif (dayCount in DictBeforeHoilday):
            x[2] = 1
        elif (dayCount in DictWorkday):
            if (x[1]==6 or ((dayCount+1) in DictHoilday)):
                x[2] = 1
        elif (x[1]==0 or x[1]==6):
            x[3] = 1
        elif (x[1]==5):
            x[2] = 1
        feature.append(x)
    return feature     
            
def setModel(clas, model):
    global modelChoose
    if (clas not in modelChoose):
        modelChoose[clas] = model
    elif (model < modelChoose[clas]):
        modelChoose[clas] = model    
    
def modelselect(trainSize, testSize, skipSize = 0):
    larclasPred = {}
    totalBias = 0
    totalCount = 0
    
    loader = dataLoader.loader("datam.csv", "lcdatam.csv")
    loader.setSize(trainSize, testSize, skipSize)
        
    # middle class
    while (True):
        midclass, trD, trL, teD, teL = loader.getNextMidClass() 
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
            simulateFeature(teD, [-2, -1])
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
                setModel(midclass, 3)
                if (larclass in larclasPred):
                    larclasPred[larclass] += teP3
                else:
                    larclasPred[larclass] = teP3
            elif (bias1 <= bias2):
                totalBias += bias1
                bias1 = math.sqrt(bias1/testSize)
                print "(Midclass %d select SARIMA, accuracy: %f)" % (midclass, bias1)
                setModel(midclass, 1)
                if (larclass in larclasPred):
                    larclasPred[larclass] += teP1
                else:
                    larclasPred[larclass] = teP1
            else:
                totalBias += bias2
                bias2 = math.sqrt(bias2/testSize)
                print "(Midclass %d select XGBOOST, accuracy: %f)" % (midclass, bias2)
                setModel(midclass, 2)
                if (larclass in larclasPred):
                    larclasPred[larclass] += teP2
                else:
                    larclasPred[larclass] = teP2
                    
    # large class
    while (True):
        larclass, trD, trL, teD, teL = loader.getNextLarClass()  
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
            simulateFeature(teD, [-2, -1])
            try:
                teP2 = xgboostPredict(array(trD), array(trL), array(teD))
            except:
                teP2 = zeros(testSize)
            
            # sum of midclasses
            try:
                teP3 = larclasPred[larclass]
            except:
                teP3 = zeros(testSize)

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
                setModel(larclass, 3)
            elif (bias1 <= bias2):
                totalBias += bias1
                bias1 = math.sqrt(bias1/testSize)
                print "(Larclass %d select SARIMA, accuracy: %f)" % (larclass, bias1)
                setModel(larclass, 1)
            else:
                totalBias += bias2
                bias2 = math.sqrt(bias2/testSize)
                print "(Larclass %d select XGBOOST, accuracy: %f)" % (larclass, bias2)
                setModel(larclass, 2)

    totalBias = math.sqrt(totalBias/totalCount)
    print "(Predict finished, accuracy: %f)" % (totalBias)        
    loader.closeFiles()
    
def submit(trainSize): 
    global larclasPred
    larclasPred = {}
    f1 = open("example.csv", "r")
    submit_csv = csv.reader(f1)
    row = submit_csv.next()
    f2 = open('submit.csv', 'wb')
    writer = csv.writer(f2)
    writer.writerow(row)
    
    loader = dataLoader.loader("datam.csv", "lcdatam.csv")
    loader.setSize(trainSize)
    
    preDate = [x for x in range(0, 9)]+[x for x in range(10, 59)]
    
    # middle class
    goal = createFeature(dt.datetime(2015,9,1), 59, 2,
                         range(31, 38), [30], [39, 40])

    while (True):
        midclass, trD, trL, teD, teL = loader.getNextMidClass()
        if (midclass == 0):
            break
        else:
            if (modelChoose[midclass] == 1):
                try:
                    model = ap.sarimaTrain(midclass, trL)
                    teP = ap.sarimaPredict(model, 59)
                except:
                    print("%d: failed to use arima, use xgboost instead" % midclass)
                    teP = xgboostPredict(array(trD), array(trL), array(goal))
            elif (modelChoose[midclass] == 2):
                teP = xgboostPredict(array(trD), array(trL), array(goal))
            else:
                teP = zeros(59)
            
            for i in preDate:
                x_int = round(teP[i])
                if (x_int < 0):
                    x_int = 0
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
    goal = createFeature(dt.datetime(2015,9,1), 59, 1,
                         range(31, 38), [30], [39, 40])

    while (True):
        larclass, trD, trL, teD, teL = loader.getNextLarClass()
        if (larclass == 0):
            break
        else:
            if (modelChoose[larclass] == 1):
                try:
                    model = ap.sarimaTrain(larclass, trL)
                    teP = ap.sarimaPredict(model, 59)
                except:
                    print("%d: failed to use arima, use xgboost instead" % larclass)
                    teP = xgboostPredict(array(trD), array(trL), array(goal))
            elif (modelChoose[larclass] == 2):
                teP = xgboostPredict(array(trD), array(trL), array(goal))
            else:
                try:
                    teP = larclasPred[larclass]
                except:
                    teP = zeros(59)

            # write file - midclass
            for i in preDate:
                x_int = round(teP[i])
                if (x_int < 0):
                    x_int = 0
                row = submit_csv.next()
                if (int(row[0]) != larclass):
                    raise KeyError
                writer.writerow([row[0], row[1], x_int])

    f1.close()
    f2.close()
    loader.closeFiles()
           
modelselect(200, 43, 0)
submit(243)