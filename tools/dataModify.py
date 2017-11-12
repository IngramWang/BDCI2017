# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 20:30:09 2017

@author: wangjun
"""

import csv
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX  
import datetime as dt

dateToModify = [34, 44, 89, 98, 105]
index = [dt.datetime(2015,1,x) for x in range(1, 32)]
index = index + [dt.datetime(2015,2,x) for x in (range(1 ,29))]
index = index + [dt.datetime(2015,3,x) for x in range(1, 32)]
index = index + [dt.datetime(2015,4,x) for x in range(1, 31)]

def getData(csvReader, count):
    data = []
    label = []
    try:
        for x in range(0, count):
            row = csvReader.next()
            data.append(row[:-1])
            label.append(int(row[-1]))
        return int(row[0]), data, label
    except StopIteration:
        return 0, [], []
    
def modifyFile(reader, writer, count):
    global dateToModify, index
    while (True):
        clas, data, label = getData(reader, count) 
        if (clas == 0):
            break     
        data0 = pd.Series(label) 
        data0.index = pd.Index(index)         
        try:
            model = SARIMAX(data0, order=(1,1,1), seasonal_order=(0,1,1,7)) 
            result = model.fit() 
        except:
            print("%d: failed to train sarimax model, abort" % clas)
            for i in range(0, count):
                writer.writerow(data[i] + [label[i]])
            continue       
        for i in dateToModify:
            label[i] = round(result.predict(i, i)[0])
        for i in range(0, count):
            writer.writerow(data[i] + [label[i]])
            
f1 = open("data.csv", "r")
reader = csv.reader(f1)
f2 = open('datam.csv', 'wb')
writer = csv.writer(f2)
modifyFile(reader, writer, 120)
f1.close()
f2.close()

f1 = open("lcdata.csv", "r")
reader = csv.reader(f1)
f2 = open('lcdatam.csv', 'wb')
writer = csv.writer(f2)
modifyFile(reader, writer, 120)
f1.close()
f2.close()

        