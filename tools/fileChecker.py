# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:55:56 2017

@author: wangjun
"""

import csv

f1 = open("submit1.csv", "r")
f1_csv = csv.reader(f1)

f2 = open("submit3.csv", "r")
f2_csv = csv.reader(f2)

lineNo = 2
row1 = f1_csv.next()
row2 = f2_csv.next()

while (True):
    try:
        row1 = f1_csv.next()
        row2 = f2_csv.next()
    except StopIteration:
        break
    if (int(row1[2])!=int(row2[2])):
        print lineNo
        i = input()
    lineNo += 1