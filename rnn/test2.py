# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:16:06 2017

@author: wangjun
"""

import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import dataLoader

import matplotlib.pyplot as plt
import xgboostPredicter

loader = dataLoader.loader("datam.csv")
loader.setSize(200, 43, 0)
midclass, trainData, trainLabel, testData, testLabel = loader.getNextMidClass()
loader.closeFiles()

seq_length = 0
data_max = 35
dataX = []
dataY = []

trainLabelN = []
for i in range(0, len(trainLabel)):
    trainLabelN.append(trainLabel[i] / data_max)

for i in range(0, len(trainLabelN) - seq_length):
    dataX.append(trainData[i+seq_length]+trainLabelN[i:i+seq_length])
    dataY.append(trainLabelN[i+seq_length])
    
X = np.reshape(dataX, (len(dataX), 1, len(trainData[0])+seq_length))
Y = np.reshape(dataY, (len(dataY), 1))

model = Sequential()
model.add(LSTM(6, input_shape=(X.shape[1], X.shape[2]), batch_size=1, stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, nb_epoch=300, batch_size=1, verbose=1)

#history = trainLabelN[-1*seq_length:]
predLabel = []
for i in range(0, len(testLabel)):
    #feature = np.array(testData[i]+history).reshape(1, 1, len(trainData[0])+seq_length)
    feature = np.array(testData[i]).reshape(1, 1, len(trainData[0]))
    predict = model.predict(feature)
    predLabel.append(predict[0][0]*data_max)
    #history.pop(0)
    #history.append(predict)
 
predLabel = np.array(predLabel)
testLabel = np.array(testLabel)
bias = sum((predLabel-testLabel)*(predLabel-testLabel))
bias = math.sqrt(bias/len(testLabel))
print(bias)
plt.plot(predLabel, color='blue',label='predict')
plt.plot(testLabel, color='red', label='origan')
plt.show(block=False)

def xgboostPredict(trainData, trainLabel, testData):
    
    xgp = xgboostPredicter.predicter()
    model = xgp.xgboostTrain(trainData, trainLabel)
    predLabel = xgp.xgboostPredict(model, testData)
    return predLabel

predLabel = xgboostPredict(trainData, trainLabel, testData)
predLabel = np.array(predLabel)
testLabel = np.array(testLabel)
bias = sum((predLabel-testLabel)*(predLabel-testLabel))
bias = math.sqrt(bias/len(testLabel))
print(bias)
plt.plot(predLabel, color='blue',label='predict')
plt.plot(testLabel, color='red', label='origan')
plt.show(block=False)
