使用随机森林
特征是day_in_week, day_in_month, holiday, discount，如果是大类再加一个 大类中中类的预测销量之和


five_fold.py是对整个数据集5折交叉，预测的结果作为新的特征，用于stacking（v6)，保存再five_fold_feature.csv