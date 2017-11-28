# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:14:17 2017

@author: wangjun
"""

# xgboostPredicter, arimaPredicter快速上手指南
# 最后更新 20171128

import arimaPredicter
import dataLoader
import xgboostPredicter

import matplotlib.pylab as plt
import datetime as dt

#读取训练数据，用其他方式读取也可以
loader = dataLoader.loader("datam.csv")
loader.setSize(200, 43, 0)
midclass, trainData, trainLabel, testData, testLabel = loader.getNextMidClass()

plt.plot(trainLabel)
plt.title('Train Label')
plt.show(block=False)

def arimaPredict():
    # 首先创建类实例
    ap = arimaPredicter.predicter()
    # 设置索引，函数的第一个参数是训练数据开始的日期，第二次参数是索引的长度，索引
    # 长度不小于训练数据的长度即可
    ap.createIndex(dt.datetime(2015,1,1), 243)
    # 可以直接调用sarimaTrain函数训练arima模型，只需将训练标签输入即可
    model = ap.sarimaTrain(trainLabel)
    # 得到模型后调用sarimaPredict函数便可以预测紧接着训练数据之后若干天的预测值，
    # 两个参数分别为先前得到的模型与预测序列的长度
    # 这是一个静态函数，可以直接通过类名调用
    predictLabel = ap.sarimaPredict(model, 43)
    # 这样便可以得到结果
    plt.plot(testLabel, color='blue',label='actual')
    plt.plot(predictLabel, color='red',label='predict')
    plt.title('ARIMA(default)')
    plt.show(block=False)
    
    # 事实上，在sarimaTrain函数中，你也可以指定ARIMA模型的两个参数（ar, ma）
    model = ap.sarimaTrain(trainLabel, para=(2, 2))
    # 如果参数指定得当，结果将更好，反之更糟糕
    predictLabel = ap.sarimaPredict(model, 43)
    plt.plot(testLabel, color='blue',label='actual')
    plt.plot(predictLabel, color='red',label='predict')
    plt.title('ARIMA(2, 2)')
    plt.show(block=False)   
    
    # 如果你不知道该指定什么参数，那么可以使用sarimaParaSelect函数选择参数，该函数
    # 的输入为类别名称，训练集，测试集及决定在参数选择时是否参考AIC的布尔变量
    # 目前来看，在参数选择时是否参考AIC的结果差不多
    # 函数执行后将会返回最优的参数以及测试集上的运行结果，同时实例中也会以类别名称为
    # 键存储这个最优参数
    para, _ = ap.sarimaParaSelect(1001, trainLabel[:-50], trainLabel[-50:], True)
    
    # 由于最优参数已被存储，再次训练是指明类别名称即可
    model = ap.sarimaTrain(trainLabel, classNo=1001)
    # 预测的方式始终相同
    predictLabel = ap.sarimaPredict(model, 43)
    plt.plot(testLabel, color='blue',label='actual')
    plt.plot(predictLabel, color='red',label='predict')
    plt.title('ARIMA(%d, %d)' % (para[0], para[1]))
    plt.show(block=False)
    
    # 需要注意的是，当模型不等收敛时，sarimaTrain函数与sarimaParaSelect函数都有可能
    # 抛出异常
    
def xgboostPredict():
    # 首先创建类实例
    xgp = xgboostPredicter.predicter()

    # 可以直接调用xgboostTrain函数训练xgboost模型，输入为训练集的特征和对应的标签
    model = xgp.xgboostTrain(trainData, trainLabel)

    # 得到模型后调用xgboostPredict函数便可以根据测试集的特征得到对应的预测值
    # 这是一个静态函数，可以直接通过类名调用
    predictLabel = xgp.xgboostPredict(model, testData)
    # 这样便可以得到结果
    plt.plot(testLabel, color='blue',label='actual')
    plt.plot(predictLabel, color='red',label='predict')
    plt.title('xgboost(default)')
    plt.show(block=False)
    
    # 在predicter类中，还有两个静态的工具函数：
    # simulateFeature函数用于将特征向量的某些位清空，如
    xgp.simulateFeature(testData, [-2, -1])
    # 可以清空测试集中所有特征向量的后两位（在我的特征定义中对应促销信息），这将使
    # 在测试集上的结果更加真实
    predictLabel = xgp.xgboostPredict(model, testData)
    plt.plot(testLabel, color='blue',label='actual')
    plt.plot(predictLabel, color='red',label='predict')
    plt.title('xgboost(default)')
    plt.show(block=False)  
    
    # createFeature函数用于创建测试用的特征向量，但只有在你的特征定义与我的一致时
    # 才能使用它
    # 其输入参数为 (开始日期,长度,后缀零数量,节假日列表,节假日前一天列表,工作日列表)
    # 列表均为对应日期的序号，从1开始计数；需要注意的是，周末自动算节假日，周五自动
    # 算节假日前一天，例如
    data = xgp.createFeature(dt.datetime(2015,9,1), 7, 1, [4], [3], [6])
    # 的输出为
    for x in data:
        print(x)
    
arimaPredict()
xgboostPredict()