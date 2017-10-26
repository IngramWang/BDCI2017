# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:48:57 2017

@author: wangjun
"""

import csv

DictHoilday = [1,2,3,48,49,50,51,52,53,54,94]
DictBeforeHoilday = [45,46,47]
midClasses = {}

date = '20150101'
dailyData = {}
# index       -0          -1          
# middle class-sales count-promotions
# string      -float      -int     
promotions = []
totalCount = 0
totalPay = 0
lineNum = 1
dayCount = 1

def writeData():
    global dailyData, promotions, totalCount, totalPay, dayCount
    day = int(date) % 100
    month = int(date) / 100 % 100
    if (month==1):
        week = (day + 3) % 7
    elif (month==2):
        week = (day - 1) % 7
    elif (month==3):
        week = (day - 1) % 7
    elif (month==4):
        week = (day + 2) % 7
    else :
        raise Exception("unknown month")
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
        beforeHoliday = 1
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
                                     totalCount, totalPay, dailyData[midclass][0]])
                else:
                    writer.writerow([midclass, dayCount, month,
                                     day, week, beforeHoliday, holiday, 
                                     0, 0,
                                     totalCount, totalPay, dailyData[midclass][0]]) 
            except ZeroDivisionError:
                pass
                #just neglect it
        for midclass in midClasses:
            if (midClasses[midclass] == 0):
                larclass = int(midclass) / 100
                if (larclass in promotionClass):
                    writer.writerow([midclass, month, dayCount,
                                     day, week, beforeHoliday, holiday, 
                                     0, promotionClass[larclass],
                                     totalCount, totalPay, 0])
                else:
                    writer.writerow([midclass, month, dayCount,
                                     day, week, beforeHoliday, holiday, 0, 0,
                                     totalCount, totalPay, 0]) 
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
        if (date != row[7]):
            writeData()
            date = row[7]
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
        
        
            
            