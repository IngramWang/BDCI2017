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

            # sarima model
            try:
                model = ap.sarimaTrain(midclass, trL, teL)
                teP1 = ap.sarimaPredict(model, testSize)
            except:
                teP1 = zeros(testSize)
            
            # xgboost model
            xgp.simulateFeature(teD, [-2, -1])
            try:
                model = xgp.xgboostTrain(trD, trL)
                teP2 = xgp.xgboostPredict(model, teD)
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
            xgp.simulateFeature(teD, [-2, -1])
            try:
                model = xgp.xgboostTrain(trD, trL)
                teP2 = xgp.xgboostPredict(model, teD)
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
    
def submit(trainSize, cvSize): 
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
    
    preDate = range(0, 9) + range(10, 59)
    
    # middle class
    goal = xgp.createFeature(dt.datetime(2015,9,1), 59, 2,
                         range(31, 38), [30], [39, 40])

    while (True):
        midclass, trD, trL, teD, teL = loader.getNextMidClass()
        if (midclass == 0):
            break
        else:
            teP = zeros(59)
            count = 3
            for i in range(0, cvSize):
                if (modelChoose[midclass][i] == 1):
                    try:
                        model = aps[i].sarimaTrain(midclass, trL)
                        teP += aps[i].sarimaPredict(model, 59)
                    except:
                        print("%d: failed to use arima" % midclass)
                        count -= 1
                elif (modelChoose[midclass][i] == 2):
                    model = xgp.xgboostTrain(trD, trL)
                    teP += xgp.xgboostPredict(model, goal)
                    
            if (count == 0):
                print("%d: failed to use arima at all, only use xgboost" % midclass)
                model = xgp.xgboostTrain(trD, trL)
                teP = xgp.xgboostPredict(model, goal)
            else:
                teP = teP / count
            
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
    goal = xgp.createFeature(dt.datetime(2015,9,1), 59, 1,
                         range(31, 38), [30], [39, 40])

    while (True):
        larclass, trD, trL, teD, teL = loader.getNextLarClass()
        if (larclass == 0):
            break
        else:           
            teP = zeros(59)
            count = 3
            for i in range(0, cvSize):
                if (modelChoose[larclass][i] == 1):
                    try:
                        model = aps[i].sarimaTrain(larclass, trL)
                        teP += aps[i].sarimaPredict(model, 59)
                    except:
                        print("%d: failed to use arima" % larclass)
                        count -= 1
                elif (modelChoose[larclass][i] == 2):
                    model = xgp.xgboostTrain(trD, trL)
                    teP += xgp.xgboostPredict(model, goal)
                elif (larclass in larclasPred):
                    teP += larclasPred[larclass]
                    
            if (count == 0):
                print("%d: failed to use arima at all, only use xgboost" % larclass)
                model = xgp.xgboostTrain(trD, trL)
                teP = xgp.xgboostPredict(model, goal)
            else:
                teP = teP / count

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
           
modelselect(aps[0], 210, 28, 5)
modelselect(aps[1], 180, 28, 35)
modelselect(aps[2], 150, 28, 65)
submit(243, 3)