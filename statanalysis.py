# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

import csv
import math

larclasPred = {}
larclasLabl = {}
totalBias = 0
totalCount = 0

temp = []

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

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import datetime as dt

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

def statanalysis():
    global larclasPred, larclasLabl, totalBias, totalCount   
    f = open("data.csv", "r")
    f_csv = csv.reader(f)
    
    midclass, trD, trL, teD, teL = getData(f_csv, 120, 0)     
    # print trL  
    data0 = pd.Series(trL)
    """
    index = [dt.datetime(2015,1,x) for x in range(1, 32)]#31
    index = index + [dt.datetime(2015,2,x) for x in (range(1,14) + range(15,29))]#27
    index = index + [dt.datetime(2015,3,x) for x in range(1, 31)]#30
    index = index + [dt.datetime(2015,4,x) for x in (range(1,9) + range(10,16) + range(17,31))]#28
    
    data0.index = pd.Index(index)
    data0[dt.datetime(2015,2,14)] = float(0)
    data0[dt.datetime(2015,3,31)] = float(0)
    data0[dt.datetime(2015,4,9)] = float(0)
    data0[dt.datetime(2015,4,16)] = float(0)
    """
    
    index = [dt.datetime(2015,1,x) for x in range(1, 32)]
    index = index + [dt.datetime(2015,2,x) for x in (range(1 ,29))]
    index = index + [dt.datetime(2015,3,x) for x in range(1, 32)]
    index = index + [dt.datetime(2015,4,x) for x in range(1, 31)]
    data0.index = pd.Index(index)

    #test_stationarity(data0)
    #print data0
   
    data1 = data0.diff(1)
    """
    data1 = data1[dt.datetime(2015,1,2):]
    test_stationarity(data1)
    
    fig = plt.figure(figsize=(12,8))
    ax1=fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data1,lags=40,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data1,lags=40,ax=ax2)
    plt.show(block=False)
    """
    model = ARIMA(data0, order=(1, 1, 1))  
    results_ARIMA = model.fit(disp=-1) 
    plt.plot(data1)
    plt.plot(results_ARIMA.fittedvalues, color='red')  
    plt.show(block=False)
    """
    for p in range(1, 5):
        for q in range(1, 5):
            try:
                model = ARIMA(data0, order=(p, 1, q))  
                results_ARIMA = model.fit()
                print p, q, results_ARIMA.aic
            except:
                pass
    """ 
    # print data0
    output = results_ARIMA.predict(1, 119)
    output1 = results_ARIMA.forecast(2)
    output2 = results_ARIMA.predict(120, 121)
    print output1
    print output2["2015-05-01"] + data0[119]
    print output2["2015-05-01"] + output2["2015-05-02"] + data0[119]
    temp.append(results_ARIMA)
    
    teL = [trL[0]]
    last = trL[0]
    for x in output:
        last = last + x
        teL.append(last)
    """
    for i in range(0, 120):
        print trL[i], teL[i]
    
    print "diff"
    
    for i in range(1, 120):
        print data1[index[i]], output[index[i]]
    """
    plt.plot(trL)
    plt.plot(teL, color='red')
    plt.show(block=False)
    
    """
    trdata = data0[:dt.datetime(2015,4,15)]
    tedata = data0[dt.datetime(2015,4,15):]
    model = ARIMA(trdata, order=(1, 1, 1)) 
    result = model.fit()
    output = result.predict(dt.datetime(2015,4,16))
    print output
    """
    
def statTest():
    global larclasPred, larclasLabl, totalBias, totalCount   
    f = open("data.csv", "r")
    f_csv = csv.reader(f)
    
    midclass, trD, trL, teD, teL = getData(f_csv, 113, 0)    
    # print trL  
    data0 = pd.Series(trL) 
    index = [dt.datetime(2015,1,x) for x in range(8, 32)]
    index = index + [dt.datetime(2015,2,x) for x in (range(1 ,29))]
    index = index + [dt.datetime(2015,3,x) for x in range(1, 32)]
    index = index + [dt.datetime(2015,4,x) for x in range(1, 31)]
    data0.index = pd.Index(index)

    trainData = data0[:dt.datetime(2015,4,15)]

    test_stationarity(trainData.diff(1)[1:])
    model = ARIMA(trainData, order=(4, 1, 4))  
    results_ARIMA = model.fit(disp=-1) 
    
    plt.plot(trainData.diff(1))
    plt.plot(results_ARIMA.fittedvalues, color='red')  
    plt.show(block=False)
    """
    pre = results_ARIMA.predict(1, 120)
    print pre
    """
    output, _, _ = results_ARIMA.forecast(trL.__len__()-trainData.__len__())
    plt.plot(trL[trainData.__len__():])
    plt.plot(output, color='red')  
    plt.show(block=False)   
    
from statsmodels.tsa.statespace.sarimax import SARIMAX    
    
def sariamTest():
    global larclasPred, larclasLabl, totalBias, totalCount   
    f = open("data.csv", "r")
    f_csv = csv.reader(f)
    
    midclass, trD, trL, teD, teL = getData(f_csv, 113, 0)    
    # print trL  
    data0 = pd.Series(trL) 
    index = [dt.datetime(2015,1,x) for x in range(8, 32)]
    index = index + [dt.datetime(2015,2,x) for x in (range(1 ,29))]
    index = index + [dt.datetime(2015,3,x) for x in range(1, 32)]
    index = index + [dt.datetime(2015,4,x) for x in range(1, 31)]
    data0.index = pd.Index(index)
    
    data1 = data0.diff(1)
    #test_stationarity(data1[8:])
    data2 = data1 - data1.shift(7)
    test_stationarity(data2[8:])

    trainData = data0[:dt.datetime(2015,4,15)]
    testData = data0[dt.datetime(2015,4,16):]

    model = SARIMAX(trainData, order=(1,1,1), seasonal_order=(0,1,1,7)) 
    result = model.fit() 
    print result.aic
    
    output = result.forecast(trL.__len__()-trainData.__len__())
    plt.plot(testData)
    plt.plot(output, color='red')  
    plt.show(block=False)
    
    output1 = result.predict()
    plt.plot(trainData)
    plt.plot(output1, color='red')  
    plt.show(block=False)
    
    bias = trainData - output1
    # plt.plot(bias)  
    # plt.show(block=False)
    print bias.__len__()
    print trainData.__len__()
    print output1.__len__()

"""    
    model = SARIMAX(trainData, order=(1,1,1), seasonal_order=(1,1,1,7)) 
    result = model.fit() 
    print result.aic
    output = result.forecast(trL.__len__()-trainData.__len__())
    plt.plot(testData)
    plt.plot(output, color='red')  
    plt.show(block=False)
""" 
sariamTest()