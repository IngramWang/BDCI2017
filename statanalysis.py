# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import datetime as dt
import pandas as pd
import numpy as np

import csv
import math
import arimaPredicter

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
            row = next(csvReader)
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
            row = next(csvReader)
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
            row = next(csvReader)
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7])]
            trainData.append(data)
            trainLabel.append(float(row[14]))
        for x in range(0, testCount):
            row = next(csvReader)
            data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                    float(row[7])]
            testData.append(data)
            testLabel.append(float(row[14]))
        return int(row[0]), trainData, trainLabel, testData, testLabel
    except StopIteration:
        return 0, [], [], [], []
    
def getBias(label, pred):
    a1 = np.array(label)
    a2 = np.array(pred)
    if (a1.__len__() != a2.__len__()):
        raise ValueError("length not equel")
    m = a1 - a2
    return math.sqrt(sum(m*m)/a1.__len__()) 
    
def sariamTest():
    f = open("datam.csv", "r")
    f_csv = csv.reader(f)
    
    # writer = open("report.txt", "w")
    
    ap = arimaPredicter.predicter();
    ap.setIndex(index)
    
    for i in range(0, 10):
        midclass, trD, trL, teD, teL = getData(f_csv, 120, 0)    
        if (midclass == 0):
            break
        
        trainData = trL[:99]
        testData = trL[99:]
        
        ap.test_stationarity(trL)
    
        greatfit = (0, 0, 0)
        minaic = 99999
        
        for p in range(0, 3):
            for q in range(0, 3):
                try:
                    ap.setPara(midclass, (p, q))
                    model = ap.sarimaTrain(midclass, trainData)
                    if (model.aic < minaic):
                        minaic = model.aic
                        greatfit = (p, 1, q)
                    result = ap.sarimaPredict(model, len(testData))
                    print("(%d,%d) %f %f\n" % (p, q, model.aic, getBias(testData, result)))
                    
                except:
                    pass
        
        print("midclass %d: %d %d\n" % (midclass, greatfit[0], greatfit[2]))       
    
    f.close()
    #writer.close()
"""    
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
"""    
sariamTest()