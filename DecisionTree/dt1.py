# coding:utf-8
from collections import Counter
import operator
import math
from _functools import reduce

'''
ID3：只适用于离散属性，对连续数据需离散化
计算熵
'''
def clacEnt(dataSet):
    classCount = Counter(sample[-1] for sample in dataSet)
    prob = [float(v) / sum(classCount.values()) for v in classCount.values()]
    return reduce(operator.add, map(lambda x: -x * math.log(x, 2)), prob)

'''
计算Gini指数。Gini(D)越小，则数据集D的纯度越高
'''
def calcGini(dataSet):
    labelCounts = Counter(sample[-1] for sample in dataSet)
    prob = [float(v) / sum(labelCounts.values()) for v in labelCounts.values()]
    return 1 - reduce(operator.add, map(lambda x: x**2, prob))