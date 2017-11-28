# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:14:17 2017

@author: wangjun
"""

# arimaPredicter快速上手指南
# 最后更新 20171128

import arimaPredicter
import dataLoader

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
    plt.title('ARIMA(1, 1)')
    plt.show(block=False)
    
    # 事实上，在sarimaTrain函数中，你也可以指定ARIMA模型的两个参数（ar, ma）
    model = ap.sarimaTrain(trainLabel, para=(2, 1))
    # 如果参数指定得当，结果将更好
    predictLabel = ap.sarimaPredict(model, 43)
    plt.plot(testLabel, color='blue',label='actual')
    plt.plot(predictLabel, color='red',label='predict')
    plt.title('ARIMA(2, 1)')
    plt.show(block=False)   
    
    # 如果你不知道该知道什么参数，那么可以使用sarimaParaSelect函数选择参数，该函数
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
    
arimaPredict()