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
    
def sarimaTrain(trainLabel):
    dataLength = trainLabel.__len__()
    data = pd.Series(trainLabel)
    index = dtIndex[0:dataLength]
    data.index = pd.Index(index)

    model = SARIMAX(data, order=(1,1,1), seasonal_order=(0,1,1,7)) 
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
    global larclasPred, larclasLabl, totalBias, totalCount   
    larclasPred = {}
    larclasLabl = {}
    totalBias = 0
    totalCount = 0
    modelChoose = []
    f = open("data.csv", "r")
    f_csv = csv.reader(f)
    
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
  
            bias1 = sum((teP1-label)*(teP1-label))
            bias2 = sum((teP2-label)*(teP2-label))
            bias3 = sum((teP3-label)*(teP3-label))
            if (bias3 < bias1 and bias3 < bias2):
                totalBias += bias3
                bias3 = math.sqrt(bias3/testSize)
                print "(Midclass %d select ZERO, accuracy: %f)" % (midclass, bias3)
                modelChoose.append(3)
                if (larclass in larclasPred):
                    larclasPred[larclass] += teP3
                else:
                    larclasPred[larclass] = teP3
            elif (bias1 < bias2):
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
               
            if (larclass in larclasLabl):
                larclasLabl[larclass] += label
            else:
                larclasLabl[larclass] = label
            #dataLog(midclass, bias, trL, teP, teL)                
   
    # print bias of large class
    for larclass in larclasPred:
        bias = sum((larclasLabl[larclass] - larclasPred[larclass])*
                   (larclasLabl[larclass] - larclasPred[larclass]))
        totalBias += bias
        totalCount += testSize
        bias = math.sqrt(bias/testSize)
        print "(Larclass %d predict finished, accuracy: %f)" % (larclass, bias)  
        
    totalBias = math.sqrt(totalBias/totalCount)
    print "(Predict finished, accuracy: %f)" % (totalBias)        
    f.close()
 
def test(trainSize, testSize):
    global larclasPred, larclasLabl, totalBias, totalCount   
    larclasPred = {}
    larclasLabl = {}
    totalBias = 0
    totalCount = 0
    f = open("data.csv", "r")
    f_csv = csv.reader(f)
    
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
            try:
                model = sarimaTrain(trL)
                teP = sarimaPredict(model, testSize)
            except:
                teP = xgboostPredict(array(trD), array(trL), array(teD))

            # count bias of midclass
            bias = 0.0
            for i in range(0, testSize):
                bias += (teP[i]-teL[i])*(teP[i]-teL[i]);
            totalBias += bias
            totalCount += testSize
            bias = math.sqrt(bias/testSize)
            print "(Midclass %d predict finished, accuracy: %f)" % (midclass, bias)
            # update bias of large class
            larclass = int(midclass/100)
            if (larclass in larclasPred):
                for i in range(0, testSize):
                    larclasPred[larclass][i] += teP[i]
                    larclasLabl[larclass][i] += teL[i]
            else:
                larclasPred[larclass] = teP
                larclasLabl[larclass] = teL
            #dataLog(midclass, bias, trL, teP, teL)                
    # print bias of large class
    for larclass in larclasPred:
        bias = 0.0
        for i in range(0, testSize):
            d = larclasLabl[larclass][i] - larclasPred[larclass][i]
            bias += d*d;
        totalBias += bias
        totalCount += testSize
        bias = math.sqrt(bias/testSize)
        print "(Larclass %d predict finished, accuracy: %f)" % (larclass, bias)  
        
    totalBias = math.sqrt(totalBias/totalCount)
    print "(Predict finished, accuracy: %f)" % (totalBias)        
    f.close()
    
def submit(trainSize): 
    global larclasPred
    larclasPred = {}
    f1 = open("data.csv", "r")
    data_csv = csv.reader(f1)
    f2 = open("submit.csv", "r")
    submit_csv = csv.reader(f2)
    submit_csv.next()
    
    # generate feature
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
            
            if (modelChoose[current] == 1):
                try:
                    model = sarimaTrain(trL)
                    teP = sarimaPredict(model, 30)
                except:
                    teP = xgboostPredict(array(trD), array(trL), array(goal))
            elif (modelChoose[current] == 2):
                teP = xgboostPredict(array(trD), array(trL), array(goal))
            else:
                teP = zeros(30)
            current += 1

            # write file - midclass
            for x in teP:
                if (x < 0):
                    x = 0
                row = submit_csv.next()
                if (int(row[0]) != midclass):
                    raise KeyError
                with open('submit1.csv', 'ab') as f:
                    writer = csv.writer(f)
                    writer.writerow([row[0], row[1], x])
            
            # count larclass
            larclass = int(midclass/100)
            if (larclass in larclasPred):
                for i in range(0, 30):
                    larclasPred[larclass][i] += teP[i]
            else:
                larclasPred[larclass] = teP  
    
    # write file - larcalss
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
      
test(106, 14)      
modelselect(106, 14)
#submit(120)