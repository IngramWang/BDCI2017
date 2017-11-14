# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:48:57 2017

@author: wangjun

用于从给定的数据集生成训练数据；
由于训练程序是按照中类顺序(而非日期顺序)训练的,生成的训练数据需使用Excel按中类
进行排序:)
"""

import csv
import datetime

DictHoilday = [1,2,3,49,50,51,52,53,54,55,96]
DictBeforeHoilday = [46,47,48]
midClasses = {}

date = datetime.datetime(2015, 1, 1)
dailyData = {}
# index       -0          -1          
# middle class-sales count-promotions
# string      -float      -int     
promotions = []
totalCount = 0
totalPay = 0
lineNum = 1
dayCount = 1

dataLog = [{}, {}, {}, {}, {}, {}, {}]

def getHistory(midclass):
    total = 0
    log = []
    for i in range(0, 7):
        try:
            temp = dataLog[i][midclass][0]
            total += temp
            log.append(temp)
        except KeyError:
            log.append(0)
    return log[0], log[1], log[2], total/7

def writeData():
    global dailyData, promotions, totalCount, totalPay, dayCount, dataLog
    day = date.day
    month = date.month
    week = (date.weekday() + 1) % 7
    if (dayCount in DictHoilday):
        holiday = 1
        beforeHoliday = 0
    elif (dayCount in DictBeforeHoilday):
        holiday = 0
        beforeHoliday = 1
    elif (week==0 or week==6):
        holiday = 1
        beforeHoliday = 0
    elif (week==5):
        holiday = 0
        beforeHoliday = 1
    else:
        holiday = 0
        beforeHoliday = 0
    promotionClass = {}
    for midclass in promotions:
        larclass = int(midclass)/100
        if larclass in promotionClass:
            promotionClass[larclass] = promotionClass[larclass] + 1;
        else:
            promotionClass[larclass] = 1;
    with open('output.csv', 'ab') as f:
        writer = csv.writer(f)
        for midclass in dailyData:
            l1, l2, l3, la = getHistory(midclass)
            if (midclass not in midClasses):
                continue
            else:
                midClasses[midclass] = 1
            try:
                larclass = int(midclass) / 100
                if (larclass in promotionClass):
                    writer.writerow([midclass, dayCount, month, 
                                     day, week, beforeHoliday, holiday, 
                                     dailyData[midclass][1],
                                     promotionClass[larclass]-dailyData[midclass][1],
                                     l1, l2, l3, la,
                                     totalCount, totalPay, dailyData[midclass][0]])
                else:
                    writer.writerow([midclass, dayCount, month,
                                     day, week, beforeHoliday, holiday, 
                                     0, 0, l1, l2, l3, la,
                                     totalCount, totalPay, dailyData[midclass][0]]) 
            except ZeroDivisionError:
                pass
                #just neglect it
        for midclass in midClasses:
            l1, l2, l3, la = getHistory(midclass)
            if (midClasses[midclass] == 0):
                larclass = int(midclass) / 100
                if (larclass in promotionClass):
                    writer.writerow([midclass, dayCount, month,
                                     day, week, beforeHoliday, holiday, 
                                     0, promotionClass[larclass],
                                     l1, l2, l3, la,
                                     totalCount, totalPay, 0])
                else:
                    writer.writerow([midclass, dayCount, month,
                                     day, week, beforeHoliday, holiday, 0, 0,
                                     l1, l2, l3, la,
                                     totalCount, totalPay, 0]) 
    dataLog.insert(0, dailyData)
    dataLog.pop()
    dailyData = {}
    promotions = []
    totalCount = 0
    totalPay = 0
    dayCount += 1
    for midclass in midClasses:
        midClasses[midclass] = 0
    
with open('submit.csv') as f:
    f_csv = csv.reader(f)
    f_csv.next()
    for row in f_csv:
        if (int(row[0]) > 100):
            midClasses[row[0]] = 0;

with open('traindata.csv') as f:
    f_csv = csv.reader(f)
    f_csv.next()
    for row in f_csv:
        lineNum += 1
        
        # check date
        day = int(row[7]) % 100
        month = int(row[7]) / 100 % 100
        tempdate = datetime.datetime(2015, month, day)
        while (date != tempdate):
            writeData()
            date = date.__add__(datetime.timedelta(1))
                
        midclass = row[3]
        if (midclass in dailyData):
            #float(row[13]) or 1
            dailyData[midclass][0] = dailyData[midclass][0]+1
            totalCount=totalCount+1
            totalPay=totalPay+float(row[14])
        else:
            dailyData[midclass] = [1, 0]
            totalCount=totalCount+1
            totalPay=totalPay+float(row[14])
        if (row[16]!='\xb7\xf1'):
            dailyData[midclass][1] = 1
            if (midclass not in promotions):
                promotions.append(midclass)
    writeData();
        
        
            
            