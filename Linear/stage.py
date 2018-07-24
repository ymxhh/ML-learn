# coding: utf-8
from numpy import *

'''
数据标准化
'''
from Linear.linear import loadDataSet
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat

'''
计算均方误差大小
'''
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

'''
逐步线性回归算法
eps:表示每次迭代需要调整的步长
'''
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    # 为了实现贪心算法建立ws的两份副本
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):  # 对每个特征
            for sign in [-1,1]:     # 分别计算增加或减少该特征对误差的影响
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                # 取最小误差
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat


if __name__ == '__main__':
    '''前向逐步线性回归'''
    abX, abY = loadDataSet('./abalone2.txt')
    stageWise(abX, abY, 0.01, 200)























