# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost as xgb
from numpy import array
import csv
import datetime as dt

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
arimaParaChoose = {}

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

def xgboostPredict(trainData, trainLabel, dataToPredict):
    dtrain = xgb.DMatrix(trainData, trainLabel)
    params = {"objective": "reg:linear"}
    gbm = xgb.train(dtrain=dtrain, params=params)
    return gbm.predict(xgb.DMatrix(dataToPredict))
    
def simData(data):
    ret = data[:]
    for i in range(0, len(ret)):
        for j in range(4, len(ret[i])):
            ret[i][j] = 0
    return ret
    
def xgboostCV(trainSize): 
    global larclasPred
    larclasPred = {}
    f1 = open("datam.csv", "r")
    data_csv = csv.reader(f1)
    f3 = open("lcdatam.csv", "r")
    lc_data_csv = csv.reader(f3)
    f4 = open('xgboost_cv.csv', 'wb')
    writer = csv.writer(f4)
    
    split = [int(trainSize/5), int(2*trainSize/5), 
             int(3*trainSize/5), int(4*trainSize/5)]
    
    while (True):
        midclass, trD, trL, teD, teL = getData(data_csv, trainSize, 0)
        if (midclass == 0):
            break
        else:
            trd1 = trD[split[0]:]
            trl1 = trL[split[0]:]
            ted1 = simData(trD[:split[0]])
            tep1 = xgboostPredict(array(trd1), array(trl1), array(ted1))
            
            trd2 = trD[:split[0]]+trD[split[1]:]
            trl2 = trL[:split[0]]+trL[split[1]:]
            ted2 = simData(trD[split[0]:split[1]])
            tep2 = xgboostPredict(array(trd2), array(trl2), array(ted2))
            
            trd3 = trD[:split[1]]+trD[split[2]:]
            trl3 = trL[:split[1]]+trL[split[2]:]
            ted3 = simData(trD[split[1]:split[2]])
            tep3 = xgboostPredict(array(trd3), array(trl3), array(ted3))
            
            trd4 = trD[:split[2]]+trD[split[3]:]
            trl4 = trL[:split[2]]+trL[split[3]:]
            ted4 = simData(trD[split[2]:split[3]])
            tep4 = xgboostPredict(array(trd4), array(trl4), array(ted4))
            
            trd5 = trD[:split[3]]
            trl5 = trL[:split[3]]
            ted5 = simData(trD[split[3]:])
            tep5 = xgboostPredict(array(trd5), array(trl5), array(ted5))
            
            ans = list(tep1) + list(tep2) + list(tep3) + list(tep4) + list(tep5)
            
            for i in range(0, trainSize):
                writer.writerow([midclass, dtIndex[i].strftime("%Y%m%d"), 
                                 ans[i]])
    
    while (True):
        larclass, trD, trL, teD, teL = getLCData(lc_data_csv, trainSize, 0)
        if (larclass == 0):
            break
        else:
            trd1 = trD[split[0]:]
            trl1 = trL[split[0]:]
            ted1 = simData(trD[:split[0]])
            tep1 = xgboostPredict(array(trd1), array(trl1), array(ted1))
            
            trd2 = trD[:split[0]]+trD[split[1]:]
            trl2 = trL[:split[0]]+trL[split[1]:]
            ted2 = simData(trD[split[0]:split[1]])
            tep2 = xgboostPredict(array(trd2), array(trl2), array(ted2))
            
            trd3 = trD[:split[1]]+trD[split[2]:]
            trl3 = trL[:split[1]]+trL[split[2]:]
            ted3 = simData(trD[split[1]:split[2]])
            tep3 = xgboostPredict(array(trd3), array(trl3), array(ted3))
            
            trd4 = trD[:split[2]]+trD[split[3]:]
            trl4 = trL[:split[2]]+trL[split[3]:]
            ted4 = simData(trD[split[2]:split[3]])
            tep4 = xgboostPredict(array(trd4), array(trl4), array(ted4))
            
            trd5 = trD[:split[3]]
            trl5 = trL[:split[3]]
            ted5 = simData(trD[split[3]:])
            tep5 = xgboostPredict(array(trd5), array(trl5), array(ted5))
            
            ans = list(tep1) + list(tep2) + list(tep3) + list(tep4) + list(tep5)
            
            for i in range(0, trainSize):
                writer.writerow([larclass, dtIndex[i].strftime("%Y%m%d"), 
                                 ans[i]])

    f1.close()
    f3.close()
    f4.close()
           
xgboostCV(120)