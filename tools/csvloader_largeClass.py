# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:48:57 2017

@author: wangjun

用于从给定的数据集生成大类训练数据；
由于训练程序是按照大类顺序(而非日期顺序)训练的,生成的训练数据需使用Excel按中类
进行排序:)
"""

import csv
import datetime

DictHoilday = [1,2,3,49,50,51,52,53,54,55,96,121,173]
DictBeforeHoilday = [46,47,48,120]
DictWorkday = [46, 58, 59]
larClasses = {}

date = datetime.datetime(2015, 1, 1)
dailyData = {}
# index       -0          -1          
# large class -sales count-promotions
# string      -float      -int     
totalCount = 0
totalPay = 0
lineNum = 1
dayCount = 1

dataLog = [{}, {}, {}, {}, {}, {}, {}]

def getHistory(larclass):
    total = 0
    log = []
    for i in range(0, 7):
        try:
            temp = dataLog[i][larclass][0]
            total += temp
            log.append(temp)
        except KeyError:
            log.append(0)
    return log[0], log[1], log[2], total/7

def writeData():
    global dailyData, totalCount, totalPay, dayCount, dataLog
    day = date.day
    month = date.month
    week = (date.weekday() + 1) % 7
    if (dayCount in DictHoilday):
        holiday = 1
        beforeHoliday = 0
    elif (dayCount in DictBeforeHoilday):
        holiday = 0
        beforeHoliday = 1
    elif (dayCount in DictWorkday):
        holiday = 0
        if (week==6 or ((dayCount+1) in DictHoilday)):
            beforeHoliday = 1
        else:
            beforeHoliday = 0
    elif (week==0 or week==6):
        holiday = 1
        beforeHoliday = 0
    elif (week==5):
        holiday = 0
        beforeHoliday = 1
    else:
        holiday = 0
        beforeHoliday = 0
    with open('lcoutput.csv', 'ab') as f:
        writer = csv.writer(f)
        for larclass in dailyData:
            l1, l2, l3, la = getHistory(larclass)
            if (larclass not in larClasses):
                continue
            else:
                larClasses[larclass] = 1
            try:
                writer.writerow([larclass, dayCount, month, 
                                 day, week, beforeHoliday, holiday, 
                                 dailyData[larclass][1],
                                 l1, l2, l3, la,
                                 totalCount, totalPay, dailyData[larclass][0]])
            except ZeroDivisionError:
                pass
                #just neglect it
        for larclass in larClasses:
            l1, l2, l3, la = getHistory(larclass)
            if (larClasses[larclass] == 0):
                writer.writerow([larclass, dayCount, month,
                                 day, week, beforeHoliday, holiday, 
                                 0,
                                 l1, l2, l3, la,
                                 totalCount, totalPay, 0])
    dataLog.insert(0, dailyData)
    dataLog.pop()
    dailyData = {}
    totalCount = 0
    totalPay = 0
    dayCount += 1
    for larclass in larClasses:
        larClasses[larclass] = 0
    
with open('example.csv') as f:
    f_csv = csv.reader(f)
    f_csv.next()
    for row in f_csv:
        if (int(row[0]) < 100):
            larClasses[row[0]] = 0;

with open('train.csv') as f:
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
                
        larclass = row[1]
        if (larclass in dailyData):
            #float(row[13]) or 1
            dailyData[larclass][0] = dailyData[larclass][0]+1
            totalCount=totalCount+1
            try:
                totalPay=totalPay+float(row[14])
            except:
                pass
        else:
            dailyData[larclass] = [1, 0]
            totalCount=totalCount+1
            try:
                totalPay=totalPay+float(row[14])
            except:
                pass
        if (row[16]!='\xb7\xf1'):
            dailyData[larclass][1] = 1
    writeData();
        
        
            
            