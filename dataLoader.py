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
            self.lar_f = open(midClassFile, "r")
            self.lar_f_csv = csv.reader(self.lar_f)
        else:
            self.lar_f = None
            self.lar_f_csv = None   
            
        
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
            self.lar_f = open(midClassFile, "r")
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

    def getNextMidClass(self):
        trainData = []
        testData = []
        trainLabel = []
        testLabel = []
        try:
            for x in range(0, self.trainCount):
                row = next(self.mid_f_csv)
                data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                        float(row[7]), float(row[8])]
                trainData.append(data)
                trainLabel.append(float(row[-1]))
                
            for x in range(0, self.testCount):
                row = next(self.mid_f_csv)
                data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                        float(row[7]), float(row[8])]
                testData.append(data)
                testLabel.append(float(row[-1]))
                
            for x in range(0, self.skipCount):  
                row = next(self.mid_f_csv)
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
                data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                        float(row[7])]
                trainData.append(data)
                trainLabel.append(float(row[-1]))
                
            for x in range(0, self.testCount):
                row = next(self.lar_f_csv)
                data = [float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                        float(row[7])]
                testData.append(data)
                testLabel.append(float(row[-1]))
                
            for x in range(0, self.skipCount):  
                row = next(self.lar_f_csv)
            return int(row[0]), trainData, trainLabel, testData, testLabel
        except StopIteration:
            return 0, [], [], [], []