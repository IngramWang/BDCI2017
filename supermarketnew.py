# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost as xgb
from numpy import array
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
    return output

def sarimaBias(model, trainLabel):
    dataLength = trainLabel.__len__()
    data = pd.Series(trainLabel)
    index = dtIndex[0:dataLength]
    data.index = pd.Index(index)
    
    pred = model.predict()
    return list(data - pred)

def test(trainSize, testSize):
    global larclasPred, larclasLabl, totalBias, totalCount   
    f = open("data.csv", "r")
    f_csv = csv.reader(f)
    
    while (True):
        midclass, trD, trL, teD, teL = getData(f_csv, trainSize, testSize)
        if (midclass == 0):
            break
        else:

            try:
                model = sarimaTrain(trL)
                bias = sarimaBias(model, trL)
                #tePM = xgboostPredict(array(trD), array(bias), array(teD))
                teP = sarimaPredict(model, testSize)
                #teP = teP + tePM
            except:
                # failed to train sariam model, just use xgboost
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
    
    while (True):
        midclass, trD, trL, teD, teL = getData(data_csv, trainSize, 0)
        if (midclass == 0):
            break
        else:
            try:
                model = sarimaTrain(trL)
                #bias = sarimaBias(model, trL)
                #tePM = xgboostPredict(array(trD), array(bias), array(teD))
                teP = sarimaPredict(model, 30)
                #teP = teP + tePM
            except:
                # failed to train sariam model, just use xgboost
                teP = xgboostPredict(array(trD), array(trL), array(goal))
                
            #teP = xgboostPredict(array(trD), array(trL), array(goal))

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
            
#test(106, 14)
submit(120)