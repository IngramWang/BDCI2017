# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:48:57 2017

@author: wangjun
"""

import csv

date = '20150101'
dailyData = {}
# index       -0          -1          -2
# middle class-sales count-sales money-promotions
# string      -float      -float      -int
promotions = []
totalCount = 0
totalPay = 0
lineNum = 1
dayCount = 1

def writeData():
    global date, dailyData, promotions, totalCount, totalPay, dayCount
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
    if (week==0 or week==6):
        holiday = 1
    else:
        holiday = 0
        #mark by hand
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
            try:
                larclass = int(midclass) / 100
                if (larclass in promotionClass):
                    writer.writerow([midclass, month, dayCount,
                                     day, week, holiday, dailyData[midclass][2],
                                     dailyData[midclass][1]/dailyData[midclass][0],
                                     promotionClass[larclass]-dailyData[midclass][2],
                                     totalCount, totalPay, dailyData[midclass][0]])
                else:
                    writer.writerow([midclass, month, dayCount,
                                     day, week, holiday, 0,
                                     dailyData[midclass][1]/dailyData[midclass][0],
                                     0,
                                     totalCount, totalPay, dailyData[midclass][0]]) 
            except ZeroDivisionError:
                pass
                #just neglect it
    dailyData = {}
    promotions = []
    totalCount = 0
    totalPay = 0
    dayCount += 1

with open('traindata.csv') as f:
    f_csv = csv.reader(f)
    f_csv.next()
    for row in f_csv:
        lineNum += 1
        if (date != row[7]):
            writeData()
            date = row[7]
        midClassNo = row[3]
        if (midClassNo in dailyData):
            dailyData[midClassNo][0] = dailyData[midClassNo][0]+float(row[13])
            dailyData[midClassNo][1] = dailyData[midClassNo][1]+float(row[14])
            totalCount=totalCount+float(row[13])
            totalPay=totalPay+float(row[14])
        else:
            dailyData[midClassNo] = [float(row[13]), float(row[14]), 0]
            totalCount=totalCount+float(row[13])
            totalPay=totalPay+float(row[14])
        if (row[16]!='\xb7\xf1'):
            dailyData[midClassNo][2] = 1
            if (midClassNo not in promotions):
                promotions.append(midClassNo)
    writeData();
        
        
            
            