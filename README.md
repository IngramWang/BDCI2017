# BDCI2017

2017年CCF大数据与计算智能大赛-小超市供销存管理优化

## 参赛人员

Wang Jun   cnwj@mail.ustc.edu.cn
Wang Fei   wf314159@mail.ustc.edu.cn

## 算法说明与结果报告

见doc文件夹下report.pdf

## 文件说明

### arimaPredicter.py 

封装后的Sarima预测器

### xgboostPredicter.py 

封装后的xgboost预测器

### dataLoader.py 

封装后的数据读取类

### data文件夹

比赛数据

train.csv 比赛给定的训练数据

example.csv 比赛给定的结果样本

datam.csv 预处理后的中类样本

lcdatam.csv 预处理后的大类样本

submit0.csv submit1.csv 比赛中提交的两个文件

### tools文件夹

用于预处理的工具

### doc文件夹

相关文档

report.pdf 实验报告

manual.py Sarima预测器与xgboost预测器的使用指南

### arima_knn文件夹

基于arima、knn的集成学习

### arima_xgboost文件夹

基于arima、xgboost的集成学习

arima_xgboost_multi.py 是实验最终用于预测的集成学习预测器

### plot_pic文件夹

销量-时间图

### rnn文件夹

基于LSTM的学习器（未封装，最终未使用）

### v3文件夹

基于随机森林的学习器（未封装，最终未使用）

### v5文件夹

基于knn的学习器（未封装，最终未使用）

### v6_stacking文件夹

基于stacking的集成学习预测器





