# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost as xgb
import numpy

train_X = numpy.array([[1],[2],[3]])
train_Y = numpy.array([[10],[20],[30]])
dtrain = xgb.DMatrix(train_X, label=train_Y)

params = {"objective": "reg:linear", "booster":"gblinear"}
gbm = xgb.train(dtrain=dtrain, params=params)
Y_pred = gbm.predict(xgb.DMatrix(numpy.array([[2.5]])))
print Y_pred