# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost as xgb
import dataLoader
import arimaPredicter

from numpy import array
from numpy import zeros
import datetime as dt
import math

dtIndex = [dt.datetime(2015,1,x) for x in range(1, 32)]
dtIndex = dtIndex + [dt.datetime(2015,2,x) for x in (range(1, 29))]
dtIndex = dtIndex + [dt.datetime(2015,3,x) for x in range(1, 32)]
dtIndex = dtIndex + [dt.datetime(2015,4,x) for x in (range(1, 31))]

def xgboostPredict(trainData, trainLabel, dataToPredict):
    dtrain = xgb.DMatrix(trainData, trainLabel)
    params = {"objective":"reg:linear", "max_depth":1, "gamma":2}
    gbm = xgb.train(dtrain=dtrain, params=params)
    return gbm.predict(xgb.DMatrix(dataToPredict))

def simulateFeature(trainData, musk):
    for feature in trainData:
        for i in musk:
            feature[i] = 0

def modelselect(trainSize, testSize, skipSize = 0):
    totalBias = 0
    totalCount = 0
    
    loader = dataLoader.loader("traindata.csv")
    loader.setSize(trainSize, testSize, skipSize)
    loader.setMidClassFeature(range(3, 9), [0, 0, 0])
    
    ap = arimaPredicter.predicter()
    ap.setIndex(dtIndex)
        
    while (True):
        midclass, trD, trL, teD, teL = loader.getNextMidClass() 
        if (midclass == 0):
            break
        else:
            """
            try:
                model = ap.sarimaTrain(midclass, trL, teL)
                output0 = model.predict()
                output1 = ap.sarimaPredict(model, testSize)
                for i in range(2, trainSize):
                    trD[i][-1] = output0[i]
                    trD[i][-2] = output0[i-1]
                    trD[i][-3] = output0[i-2]
                teD[0][-1] = output1[0]
                teD[0][-2] = output0[-1]
                teD[0][-3] = output0[-2]
                teD[1][-1] = output1[1]
                teD[1][-2] = output1[0]
                teD[1][-3] = output0[-1]
                for i in range(2, testSize):
                    teD[i][-1] = output1[i]
                    teD[i][-2] = output1[i-1]
                    teD[i][-3] = output1[i-2]
            except:
                pass
            """   
            
            # xgboost model
            simulateFeature(teD, [-5, -4])
            try:
                teP2 = xgboostPredict(array(trD), array(trL), array(teD))
            except:
                teP2 = zeros(testSize)

            # count bias of midclass
            label = array(teL)
            totalCount += testSize

            bias2 = sum((teP2-label)*(teP2-label))
            totalBias += bias2
            bias2 = math.sqrt(bias2/testSize)
            print("(Midclass %d select XGBOOST, accuracy: %f)" % (midclass, bias2))

    totalBias = math.sqrt(totalBias/totalCount)
    print("(Predict finished, accuracy: %f)" % (totalBias))       
    loader.closeFiles()
           
modelselect(75, 14, 31)