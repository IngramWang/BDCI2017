# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost as xgb
from numpy import array
from numpy import zeros
import csv
import math

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX  
import statsmodels.api as sm
import datetime as dt
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller 

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

def getData(csvReader, trainCount, testCount):
    trainData = []
    testData = []
    trainLabel = []
    testLabel = []
    try:
        for x in range(0, trainCount):
            row = csvReader.next()
            """
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8]), float(row[9]), float(row[10]),
                    float(row[11]), float(row[12])]
            """
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8])]
            trainData.append(data)
            trainLabel.append(float(row[15]))
        for x in range(0, testCount):
            row = csvReader.next()
            """
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8]), float(row[9]), float(row[10]),
                    float(row[11]), float(row[12])]
            """
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8])]
            testData.append(data)
            testLabel.append(float(row[15]))
        return int(row[0]), trainData, trainLabel, testData, testLabel
    except StopIteration:
        return 0, [], [], [], []
    
def getLCData(csvReader, trainCount, testCount):
    trainData = []
    testData = []
    trainLabel = []
    testLabel = []
    try:
        for x in range(0, trainCount):
            row = csvReader.next()
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7])]
            trainData.append(data)
            trainLabel.append(float(row[14]))
        for x in range(0, testCount):
            row = csvReader.next()
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7])]
            testData.append(data)
            testLabel.append(float(row[14]))
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

def xgboostPredict(trainData, trainLabel, dataToPredict):
    dtrain = xgb.DMatrix(trainData, trainLabel)
    params = {"objective": "reg:linear"}
    gbm = xgb.train(dtrain=dtrain, params=params)
    return gbm.predict(xgb.DMatrix(dataToPredict))

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    
    #Get AR and MA parameter
    fig = plt.figure(figsize=(12,8))
    ax1=fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(timeseries, lags=20, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(timeseries, lags=20, ax=ax2)
    plt.show(block=False)
    
def sarimaTrain(trainLabel, ar=1, ma=1, sar=0, sma=1):
    dataLength = trainLabel.__len__()
    data = pd.Series(trainLabel)
    index = dtIndex[0:dataLength]
    data.index = pd.Index(index)

    model = SARIMAX(data, order=(ar,1,ma), seasonal_order=(sar,1,sma,7)) 
    return model.fit() 

def sarimaPredict(model, predictLength):
    output = model.forecast(predictLength)
    return array(output)

def sarimaBias(model, trainLabel):
    dataLength = trainLabel.__len__()
    data = pd.Series(trainLabel)
    index = dtIndex[0:dataLength]
    data.index = pd.Index(index)
    
    pred = model.predict()
    """
    plt.plot(data, color='blue',label='Original')
    plt.plot(pred, color='red', label='Predicted')
    plt.show(block=False)
    """
    return list(data - pred)

def modelselect(trainSize, testSize):
    global larclasPred, totalBias, totalCount, modelChoose, lcModelChoose 
    larclasPred = {}
    totalBias = 0
    totalCount = 0
    modelChoose = []
    lcModelChoose = []
    f = open("datam.csv", "r")
    f_csv = csv.reader(f)
    lc_f = open("lcdatam.csv", "r")
    lc_f_csv = csv.reader(lc_f)
        
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
        midclass, trD, trL, _, teL = getData(f_csv, trainSize, testSize)   
        if (midclass == 0):
            break
        else:

            # sarima model
            try:
                model = sarimaTrain(trL)
                teP0 = sarimaPredict(model, testSize, 0, 1, 0, 1)
            except:
                teP0 = zeros(testSize)
                
            try:
                model = sarimaTrain(trL)
                teP1 = sarimaPredict(model, testSize)
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
  
            bias0 = sum((teP0-label)*(teP0-label))
            bias1 = sum((teP1-label)*(teP1-label))
            bias2 = sum((teP2-label)*(teP2-label))
            bias3 = sum((teP3-label)*(teP3-label))
            if (bias3 < bias0 and bias3 < bias1 and bias3 < bias2):
                totalBias += bias3
                bias3 = math.sqrt(bias3/testSize)
                print "(Midclass %d select ZERO, accuracy: %f)" % (midclass, bias3)
                modelChoose.append(3)
                if (larclass in larclasPred):
                    larclasPred[larclass] += teP3
                else:
                    larclasPred[larclass] = teP3
            elif (bias0 < bias1 and bias0 < bias2):
                totalBias += bias0
                bias0 = math.sqrt(bias0/testSize)
                print "(Midclass %d select SARIMA[0], accuracy: %f)" % (midclass, bias0)
                modelChoose.append(0)
                if (larclass in larclasPred):
                    larclasPred[larclass] += teP0
                else:
                    larclasPred[larclass] = teP0
            elif (bias1 < bias2):
                totalBias += bias1
                bias1 = math.sqrt(bias1/testSize)
                print "(Midclass %d select SARIMA[1], accuracy: %f)" % (midclass, bias1)
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
        larclass, trD, trL, _, teL = getLCData(lc_f_csv, trainSize, testSize)   
        if (larclass == 0):
            break
        else:

            # sarima model
            try:
                model = sarimaTrain(trL)
                teP0 = sarimaPredict(model, testSize, 0, 1, 0, 1)
            except:
                teP0 = zeros(testSize)
                
            try:
                model = sarimaTrain(trL)
                teP1 = sarimaPredict(model, testSize)
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
  
            bias0 = sum((teP0-label)*(teP0-label))
            bias1 = sum((teP1-label)*(teP1-label))
            bias2 = sum((teP2-label)*(teP2-label))
            bias3 = sum((teP3-label)*(teP3-label))
            if (bias3 < bias0 and bias3 < bias1 and bias3 < bias2):
                totalBias += bias3
                bias3 = math.sqrt(bias3/testSize)
                print "(Larclass %d select SUM, accuracy: %f)" % (larclass, bias3)
                lcModelChoose.append(3)
            elif (bias0 < bias1 and bias0 < bias2):
                totalBias += bias0
                bias0 = math.sqrt(bias0/testSize)
                print "(Larclass %d select SARIMA[0], accuracy: %f)" % (larclass, bias0)
                lcModelChoose.append(0)
            elif (bias1 < bias2):
                totalBias += bias1
                bias1 = math.sqrt(bias1/testSize)
                print "(Larclass %d select SARIMA[1], accuracy: %f)" % (larclass, bias1)
                lcModelChoose.append(1)
            else:
                totalBias += bias2
                bias2 = math.sqrt(bias2/testSize)
                print "(Larclass %d select XGBOOST, accuracy: %f)" % (larclass, bias2)
                lcModelChoose.append(2)

    totalBias = math.sqrt(totalBias/totalCount)
    print "(Predict finished, accuracy: %f)" % (totalBias)        
    f.close()
    lc_f.close()
    
def submit(trainSize): 
    global larclasPred
    larclasPred = {}
    f1 = open("datam.csv", "r")
    data_csv = csv.reader(f1)
    f2 = open("submit.csv", "r")
    submit_csv = csv.reader(f2)
    submit_csv.next()
    f3 = open("lcdatam.csv", "r")
    lc_data_csv = csv.reader(f3)
    
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
        midclass, trD, trL, teD, teL = getData(data_csv, trainSize, 0)
        if (midclass == 0):
            break
        else:
            if (modelChoose[current] == 0):
                try:
                    model = sarimaTrain(trL, 0, 1, 0, 1)
                    teP = sarimaPredict(model, 30)
                except:
                    print("%d: failed to use arima, use xgboost instead" % midclass)
                    teP = xgboostPredict(array(trD), array(trL), array(goal))
            elif (modelChoose[current] == 1):
                try:
                    model = sarimaTrain(trL)
                    teP = sarimaPredict(model, 30)
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
                with open('submit2.csv', 'ab') as f:
                    writer = csv.writer(f)
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
        larclass, trD, trL, teD, teL = getLCData(lc_data_csv, trainSize, 0)
        if (larclass == 0):
            break
        else:
            if (lcModelChoose[current] == 0):
                try:
                    model = sarimaTrain(trL, 0, 1, 0, 1)
                    teP = sarimaPredict(model, 30)
                except:
                    print("%d: failed to use arima, use xgboost instead" % larclass)
                    teP = xgboostPredict(array(trD), array(trL), array(goal))
            elif (lcModelChoose[current] == 1):
                try:
                    model = sarimaTrain(trL)
                    teP = sarimaPredict(model, 30)
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
                with open('submit2.csv', 'ab') as f:
                    writer = csv.writer(f)
                    writer.writerow([row[0], row[1], x_int])

    f1.close()
    f2.close()
    f3.close()
           
modelselect(99, 21)
submit(120)