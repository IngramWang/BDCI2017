# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import datetime as dt
import numpy as np

import csv
import arimaPredicter
import dataLoader

index = [dt.datetime(2015,1,x) for x in range(1, 32)]
index = index + [dt.datetime(2015,2,x) for x in (range(1 ,29))]
index = index + [dt.datetime(2015,3,x) for x in range(1, 32)]
index = index + [dt.datetime(2015,4,x) for x in range(1, 31)]
    
def sariamOutput():
    loader = dataLoader.loader("datam.csv", "lcdatam.csv")
    loader.setSize(90, 0, 30)
    
    f1 = open("result01.csv", "wb")
    writer1 = csv.writer(f1)
    f2 = open("result11.csv", "wb")
    writer2 = csv.writer(f2)
    f3 = open("result12.csv", "wb")
    writer3 = csv.writer(f3)
    
    ap = arimaPredicter.predicter();
    ap.setIndex(index)
    
    while (True):
        midclass, _, trainData, _, _ = loader.getNextMidClass()  
        if (midclass == 0):
            break
        
        ap.setPara(midclass, (0, 1))
        try:
            model = ap.sarimaTrain(midclass, trainData)
            result = ap.sarimaPredict(model, 30)
        except:
            result = np.zeros(30)
        for i in range(0, 30):
            writer1.writerow([midclass, "201504%02d" % (i+1), result[i]])
        
    
        ap.setPara(midclass, (1, 1))
        try:
            model = ap.sarimaTrain(midclass, trainData)
            result = ap.sarimaPredict(model, 30)
        except:
            result = np.zeros(30)
        for i in range(0, 30):
            writer2.writerow([midclass, "201504%02d" % (i+1), result[i]])
            
        ap.setPara(midclass, (1, 2))
        try:
            model = ap.sarimaTrain(midclass, trainData)
            result = ap.sarimaPredict(model, 30)
        except:
            result = np.zeros(30)
        for i in range(0, 30):
            writer3.writerow([midclass, "201504%02d" % (i+1), result[i]])
    
    
    while (True):
        larclass, _, trainData, _, _ = loader.getNextLarClass()
        if (larclass == 0):
            break
        
        ap.setPara(larclass, (0, 1))
        try:
            model = ap.sarimaTrain(larclass, trainData)
            result = ap.sarimaPredict(model, 30)
        except:
            result = np.zeros(30)
        for i in range(0, 30):
            writer1.writerow([larclass, "201504%02d" % (i+1), result[i]])
        
    
        ap.setPara(larclass, (1, 1))
        try:
            model = ap.sarimaTrain(larclass, trainData)
            result = ap.sarimaPredict(model, 30)
        except:
            result = np.zeros(30)
        for i in range(0, 30):
            writer2.writerow([larclass, "201504%02d" % (i+1), result[i]])
            
        ap.setPara(larclass, (1, 2))
        try:
            model = ap.sarimaTrain(larclass, trainData)
            result = ap.sarimaPredict(model, 30)
        except:
            result = np.zeros(30)
        for i in range(0, 30):
            writer3.writerow([larclass, "201504%02d" % (i+1), result[i]])
            
    f1.close()
    f2.close()
    f3.close()
    loader.closeFiles()

sariamOutput()