# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import datetime as dt
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

index = [dt.datetime(2015,1,x) for x in range(1, 32)]
index = index + [dt.datetime(2015,2,x) for x in (range(1 ,29))]
index = index + [dt.datetime(2015,3,x) for x in range(1, 32)]
index = index + [dt.datetime(2015,4,x) for x in range(1, 31)]

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

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

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
    
def getBias(label, pred):
    a1 = np.array(label)
    a2 = np.array(pred)
    if (a1.__len__() != a2.__len__()):
        raise ValueError("length not equel")
    m = a1 - a2
    return math.sqrt(sum(m*m)/a1.__len__())
    
from statsmodels.tsa.statespace.sarimax import SARIMAX    
    
def sariamTest():
    global larclasPred, larclasLabl, totalBias, totalCount   
    f = open("datam.csv", "r")
    f_csv = csv.reader(f)
    
    writer = open("report.txt", "w")
    
    while (True):
        midclass, trD, trL, teD, teL = getData(f_csv, 120, 0)    
        if (midclass == 0):
            break
        # print trL  
        data0 = pd.Series(trL) 
        data0.index = pd.Index(index)
        
        trainData = data0[:dt.datetime(2015,4,9)]
        testData = data0[dt.datetime(2015,4,10):]
    
        greatfit = (0, 0, 0)
        minaic = 99999
        
        for p in range(0, 3):
            for q in range(0, 3):
                try:
                    model = SARIMAX(trainData, order=(p,1,q), seasonal_order=(0,1,1,7)) 
                    result = model.fit() 
                    if (result.aic < minaic):
                        minaic = result.aic
                        greatfit = (p, 1, q)
                    
                    output = result.forecast(trL.__len__()-trainData.__len__())
                    """    
                    plt.plot(testData)
                    plt.plot(output, color='red')  
                    plt.show(block=False)
                    """
                    writer.writelines("(%d,%d) %f %f\n" % (p, q, result.aic, getBias(testData, output)))
                    
                except:
                    pass
        
        writer.writelines("midclass %d: %d %d\n" % (midclass, greatfit[0], greatfit[2]))       
    
    f.close()
    writer.close()
    
def test_Ljung_Box(timeseries, l):
    acf, q, p = sm.tsa.acf(timeseries, nlags=l, qstat=True)
    out = np.c_[range(1, l+1), acf[1:], q, p]
    output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
    output = output.set_index('lag')
    print output
    
import arch    
    
def sariamGarchTest():
    global larclasPred, larclasLabl, totalBias, totalCount, temp
    f = open("datam.csv", "r")
    f_csv = csv.reader(f)
    
    for i in range(0, 1):
        midclass, trD, trL, teD, teL = getData(f_csv, 120, 0)    
        if (midclass == 0):
            break
        # print trL  
        data0 = pd.Series(trL) 
        data0.index = pd.Index(index)
        
        trainData = data0[:dt.datetime(2015,4,9)]
        testData = data0[dt.datetime(2015,4,10):]
    
        model = SARIMAX(trainData, order=(1,1,1), seasonal_order=(0,1,1,7)) 
        result = model.fit() 
        
        at = trainData - result.fittedvalues
        #plt.plot(at, color='red')  
        #plt.show(block=False)    
        
        at2 = np.square(at)
        plt.plot(at2, color='red')  
        plt.show(block=False)  
        #test_Ljung_Box(at2, 10)
        
        amodel = arch.arch_model(at2) 
        aresult = amodel.fit(disp='off')
        aresult.summary()
        temp.append(aresult)
        output1 = result.forecast(trL.__len__()-trainData.__len__())
        forecasts = aresult.forecast(horizon=5, start=dt.datetime(2015,4,9))
        print forecasts.mean[dt.datetime(2015,4,9):]
        print forecasts.variance[dt.datetime(2015,4,9):]
    f.close()
    
sariamGarchTest()