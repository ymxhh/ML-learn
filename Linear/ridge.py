# coding: utf-8
from numpy import *

'''
岭回归
'''
from Linear.linear import loadDataSet

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

'''用于在一组lambda上测试结果'''
def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMeans = mean(yMat, 0)   # 按行取mean
    yMat = yMat - yMeans # 数据标准化
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0) # 方差
    xMat = (xMat - xMeans) / xVar   # 所有特征减去各自的均值并除以方差
    numTestPts = 30 # 取30个不同的lambda调用函数
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i,:] = ws.T
    return wMat

if __name__ == '__main__':
    '''岭回归'''
    abX, abY = loadDataSet('./abalone2.txt')
    ridgeWeights = ridgeTest(abX, abY)  # 得到30组回归系数
    # 缩减效果图
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
    '''
    横轴表示第i组数据，纵轴表示该组数据对应的回归系数值'''
    