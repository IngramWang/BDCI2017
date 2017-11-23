# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:25:37 2017

@author: wangjun
"""

from numpy import array
from numpy import log
from numpy import exp
import math

import datetime as dt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX  
import statsmodels.api as sm

import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller 

class predicter():
    def __init__(self):
        self.ParaChoose = {}
        self.dtIndex = []
        
    def setIndex(self, index):
        self.dtIndex = index[:]
        
    def getIndex(self):
        return self.dtIndex
    
    def createIndex(self, date_from, length):
        delta = dt.timedelta(days=1)
        now = date_from
        self.dtIndex = []
        for i in range(0, length):
            self.dtIndex.append(now)
            now = now + delta
        return self.dtIndex

    def setPara(self, clas, para):
        if (type(para)!=tuple or len(para)!=2):
            raise TypeError("timeserise should be (ar, ma)")
        self.ParaChoose[clas] = para
    
    def getPara(self):
        return self.ParaChoose

    def test_stationarity(self, timeseries):
        #Determing rolling statistics
        if (type(timeseries) == list):
            length = len(timeseries)
            timeseries = pd.Series(timeseries)
            timeseries.index = pd.Index(self.dtIndex[0:length])
        elif (type(timeseries) != pd.core.series.Series):
            raise TypeError("timeserise should be a list or series")
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
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
        
        #Get AR and MA parameter
        fig = plt.figure(figsize=(12,8))
        ax1=fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(timeseries, lags=20, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(timeseries, lags=20, ax=ax2)
        plt.show(block=False)
    
    def sarimaTrain(self, classNo, trainLabel, testLabel=[]):
        dataLength = trainLabel.__len__()
        data = pd.Series(trainLabel)
        for i in range(0, dataLength):
            data[i] = log(data[i] + 1)
        index = self.dtIndex[0:dataLength]
        data.index = pd.Index(index)
        
        if (testLabel.__len__() == 0):
            try:
                (ar, ma) = self.ParaChoose[classNo]
            except KeyError:
                print("%d: no pre-trained parameter, use (1,1) default" % classNo)
                (ar, ma) = (1, 1)
            return SARIMAX(data, order=(ar,1,ma), seasonal_order=(0,1,1,7)).fit()
        else:
            minBias = 99999.0
            minAic = 99999.0
            (ar, ma) = (0, 0)
            label = array(testLabel)
            for p, q in [(1, 1), (0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
                try:
                    model = SARIMAX(data, order=(p,1,q), seasonal_order=(0,1,1,7)).fit()
                    output = array(model.forecast(testLabel.__len__()))       
                    for i in range(0, len(testLabel)):
                        output[i] = exp(output[i]) - 1
                    bias = math.sqrt(sum((output-label)*(output-label))/testLabel.__len__())
                    if (bias < minBias and model.aic < minAic):
                        (ar, ma) = (p, q)
                        minBias = bias
                        minAic = model.aic
                        bestModel = model
                except:
                    pass
            if (minBias < 90000.0):
                self.ParaChoose[classNo] = (ar, ma)
                return bestModel
            else:
                raise ValueError

    def sarimaPredict(self, model, predictLength):
        output = model.forecast(predictLength)
        for i in range(0, predictLength):
            output[i] = exp(output[i]) - 1
        return array(output)

    def checkBias(self, model, trainLabel):
        dataLength = trainLabel.__len__()
        data = pd.Series(trainLabel)
        index = self.dtIndex[0:dataLength]
        data.index = pd.Index(index)
        
        pred = model.predict()
        plt.plot(data, color='blue',label='Original')
        plt.plot(pred, color='red', label='Predicted')
        plt.show(block=False)
        return list(data - pred)
