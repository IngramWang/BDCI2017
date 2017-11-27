# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:53:16 2017

@author: wangjun
"""

import xgboost as xgb
from numpy import array
import datetime as dt
import numpy

class predicter:
    def __init__(self, params = {"objective":"reg:linear", "max_depth":1, "gamma":2}):
        self.params = params
  
    def setDefaultParams(self, params):
        self.params = params
    
    def xgboostTrain(self, trainData, trainLabel, params = {}):
        if (type(trainData)!=numpy.ndarray):
            trainData = array(trainData)
        if (type(trainLabel)!=numpy.ndarray):
            trainLabel = array(trainLabel)
        dTrain = xgb.DMatrix(trainData, trainLabel)
        if (len(params)==0):
            params = self.params
        model = xgb.train(dtrain=dTrain, params=params)
        return model
    
    @staticmethod
    def xgboostPredict(model, dataToPredict):
        if (type(dataToPredict)!=numpy.ndarray):
            dataToPredict = array(dataToPredict)
        return model.predict(xgb.DMatrix(dataToPredict))    

    @staticmethod
    def simulateFeature(trainData, musk):
        for feature in trainData:
            for i in musk:
                feature[i] = 0
             
    @staticmethod
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