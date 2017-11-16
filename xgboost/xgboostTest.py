# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost as xgb
import dataLoader

from numpy import array
from numpy import zeros
import math

def xgboostPredict(trainData, trainLabel, dataToPredict):
    dtrain = xgb.DMatrix(trainData, trainLabel)
    params = {"objective": "reg:linear"}
    gbm = xgb.train(dtrain=dtrain, params=params)
    return gbm.predict(xgb.DMatrix(dataToPredict))

def modelselect(trainSize, testSize, skipSize = 0):
    totalBias = 0
    totalCount = 0
    
    loader = dataLoader.loader("traindata.csv")
    loader.setSize(trainSize, testSize, skipSize)
        
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
        midclass, trD, trL, _, teL = loader.getNextMidClass() 
        if (midclass == 0):
            break
        else:
            
            # xgboost model
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