# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import arimaPredicter
import dataLoader
import xgboostPredicter

from numpy import array
from numpy import zeros
import csv
import math
import datetime as dt

aps = []
for i in range(0, 3):
    ap = arimaPredicter.predicter()
    ap.createIndex(dt.datetime(2015,1,1), 243)
    aps.append(ap)

xgp = xgboostPredicter.predicter()

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
            
def setModel(clas, model):
    global modelChoose
    if (clas not in modelChoose):
        modelChoose[clas] = [model]
    elif (model < modelChoose[clas]):
        modelChoose[clas].append(model)
        
def trainAndCompare(ap, clas, trD, trL, teD, teL, teP3):
    testSize = len(teL)
    # sarima model
    try:
        (_, teP1) = ap.sarimaParaSelect(clas, trL, teL)
    except:
        teP1 = zeros(testSize)
            
    # xgboost model
    xgp.simulateFeature(teD, [-2, -1])
    try:
        model = xgp.xgboostTrain(trD, trL)
        teP2 = xgp.xgboostPredict(model, teD)
    except:
        teP2 = zeros(testSize)
        
    label = array(teL)
    bias1 = sum((teP1-label)*(teP1-label))
    bias2 = sum((teP2-label)*(teP2-label))
    bias3 = sum((teP3-label)*(teP3-label))
    if (bias3 <= bias1 and bias3 <= bias2):
        return (3, bias3, teP3)
    elif (bias1 <= bias2):
        return (1, bias1, teP1)
    else:
        return (2, bias2, teP2)
    
def modelselect(ap, trainSize, testSize, skipSize = 0):
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
            (model, bias, teP) = trainAndCompare(ap, midclass, trD, trL, teD, teL, zeros(testSize))

            larclass = int(midclass/100)
            totalCount += testSize
            totalBias += bias
            bias = math.sqrt(bias/testSize)
            print("(Midclass %d select model %d, accuracy: %f)" % (midclass, model, bias))
            setModel(midclass, model)
            if (larclass in larclasPred):
                larclasPred[larclass] += teP
            else:
                larclasPred[larclass] = teP
                    
    # large class
    while (True):
        larclass, trD, trL, teD, teL = loader.getNextLarClass()  
        if (larclass == 0):
            break
        else:
            if (larclass in larclasPred):
                (model, bias, teP) = trainAndCompare(ap, larclass, trD, trL, teD, teL, larclasPred[larclass])
            else:
                (model, bias, teP) = trainAndCompare(ap, larclass, trD, trL, teD, teL, zeros(testSize))

            totalCount += testSize            
            totalBias += bias
            bias = math.sqrt(bias/testSize)
            print("(Larclass %d select model %d, accuracy: %f)" % (larclass, model, bias))
            setModel(larclass, model)

    totalBias = math.sqrt(totalBias/totalCount)
    print("(Predict finished, accuracy: %f)" % (totalBias))       
    loader.closeFiles()
    
def writeClass(clas, result, dates, checker, writer):
    for i in dates:
        x_int = round(result[i])
        if (x_int < 0):
            x_int = 0
        row = checker.next()
        if (int(row[0]) != clas):
            raise KeyError
        writer.writerow([row[0], row[1], x_int])
        
def predictClass(clas, cvSize, trD, trL, teD, teP3):
    teP = zeros(59)
    count = cvSize
    for i in range(0, cvSize):
        if (modelChoose[clas][i] == 1):
            try:
                model = aps[i].sarimaTrain(trL, clas)
                teP += aps[i].sarimaPredict(model, 59)
            except:
                print("%d: failed to use arima" % clas)
                count -= 1
        elif (modelChoose[clas][i] == 2):
            model = xgp.xgboostTrain(trD, trL)
            teP += xgp.xgboostPredict(model, teD)
        else:
            teP += teP3
                    
    if (count == 0):
        print("%d: failed to use arima at all, only use xgboost" % clas)
        model = xgp.xgboostTrain(trD, trL)
        teP = xgp.xgboostPredict(model, teD)
    else:
        teP = teP / count
    return teP
    
    
def submit(trainSize, cvSize): 
    larclasPred = {}
    f1 = open("example.csv", "r")
    submit_csv = csv.reader(f1)
    row = submit_csv.next()
    f2 = open('submit.csv', 'wb')
    writer = csv.writer(f2)
    writer.writerow(row)
    
    loader = dataLoader.loader("datam.csv", "lcdatam.csv")
    loader.setSize(trainSize)
    
    preDate = range(0, 9) + range(10, 59)
    
    # middle class
    goal = xgp.createFeature(dt.datetime(2015,9,1), 59, 2,
                         range(31, 38), [30], [39, 40])

    while (True):
        midclass, trD, trL, teD, teL = loader.getNextMidClass()
        if (midclass == 0):
            break
        else:
            teP = predictClass(midclass, cvSize, trD, trL, goal, zeros(59))
            writeClass(midclass, teP, preDate, submit_csv, writer)
            
            # count larclass
            larclass = int(midclass/100)
            if (larclass in larclasPred):
                larclasPred[larclass] += teP
            else:
                larclasPred[larclass] = teP  
    
    # large class
    goal = xgp.createFeature(dt.datetime(2015,9,1), 59, 1,
                         range(31, 38), [30], [39, 40])

    while (True):
        larclass, trD, trL, teD, teL = loader.getNextLarClass()
        if (larclass == 0):
            break
        else:           
            if (larclass in larclasPred):
                teP = predictClass(larclass, cvSize, trD, trL, goal, larclasPred[larclass])
            else:
                teP = predictClass(larclass, cvSize, trD, trL, goal, zeros(59))
            writeClass(larclass, teP, preDate, submit_csv, writer)

    f1.close()
    f2.close()
    loader.closeFiles()
           
modelselect(aps[0], 210, 28, 5)
modelselect(aps[1], 180, 28, 35)
modelselect(aps[2], 150, 28, 65)
submit(243, 3)