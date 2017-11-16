# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv

class loader():
    def __init__(self, midClassFile = "", LarClassFile = ""):
        if (midClassFile != ""):
            self.mid_f = open(midClassFile, "r")
            self.mid_f_csv = csv.reader(self.mid_f)
        else:
            self.mid_f = None
            self.mid_f_csv = None
        if (LarClassFile != ""):
            self.lar_f = open(LarClassFile, "r")
            self.lar_f_csv = csv.reader(self.lar_f)
        else:
            self.lar_f = None
            self.lar_f_csv = None
        self.trainCount = 120
        self.testCount = 0
        self.skipCount = 0
        self.midClassFeature = range(3, 9)
        self.midSuffix = []
        self.larClassFeature = range(3, 8)
        self.larSuffix = []
            
        
    def setFile(self, midClassFile = "", LarClassFile = ""):
        if (midClassFile != ""):
            try:
                self.mid_f.close()
            except:
                pass
            self.mid_f = open(midClassFile, "r")
            self.mid_f_csv = csv.reader(self.mid_f)
        if (LarClassFile != ""):
            try:
                self.lar_f.close()
            except:
                pass
            self.lar_f = open(LarClassFile, "r")
            self.lar_f_csv = csv.reader(self.lar_f)
            
    def closeFiles(self):
        try:
            self.mid_f.close()
        except:
            pass
        try:
            self.lar_f.close()
        except:
            pass

    def setSize(self, train, test = 0, skip = 0):
        self.trainCount = train
        self.testCount = test
        self.skipCount = skip
        
    def setMidClassFeature(self, feature=[], suffix=[]):
        self.midClassFeature = feature
        self.midSuffix = suffix
        
    def setLarClassFeature(self, feature=[], suffix=[]):
        self.larClassFeature = feature
        self.larSuffix = suffix

    def getNextMidClass(self):
        trainData = []
        testData = []
        trainLabel = []
        testLabel = []
        try:
            for x in range(0, self.trainCount):
                row = next(self.mid_f_csv)
                data = []
                for y in self.midClassFeature:
                    data.append(float(row[y]))
                data = data + self.midSuffix
                trainData.append(data)
                trainLabel.append(float(row[-1]))
                
            for x in range(0, self.testCount):
                row = next(self.mid_f_csv)
                data = []
                for y in self.midClassFeature:
                    data.append(float(row[y]))
                data = data + self.midSuffix
                testData.append(data)
                testLabel.append(float(row[-1]))
                
            for x in range(0, self.skipCount):  
                next(self.mid_f_csv)
            return int(row[0]), trainData, trainLabel, testData, testLabel
        except StopIteration:
            return 0, [], [], [], []
        
    def getNextLarClass(self):
        trainData = []
        testData = []
        trainLabel = []
        testLabel = []
        try:
            for x in range(0, self.trainCount):
                row = next(self.lar_f_csv)
                data = []
                for y in self.larClassFeature:
                    data.append(float(row[y]))
                data = data + self.larSuffix
                trainData.append(data)
                trainLabel.append(float(row[-1]))
                
            for x in range(0, self.testCount):
                row = next(self.lar_f_csv)
                data = []
                for y in self.larClassFeature:
                    data.append(float(row[y]))
                data = data + self.larSuffix
                testData.append(data)
                testLabel.append(float(row[-1]))
                
            for x in range(0, self.skipCount):  
                next(self.lar_f_csv)
            return int(row[0]), trainData, trainLabel, testData, testLabel
        except StopIteration:
            return 0, [], [], [], []