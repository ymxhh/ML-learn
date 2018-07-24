# coding: utf-8
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('./testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 为方便计算将x0设为1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1+exp(-inX))
      
'''
梯度上升算法
dataMatIn    2维numpy数组（100X3）
classLabels    类标签（1x100）
'''                  
def gradAscent(dataMatIn, classLabels):
    # 将输入转换为numpy矩阵的数据类型
    dataMatrix = mat(dataMatIn)
    # transpose() 转置
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    # 向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

'''
随机梯度上升法
'''
def stocGradAscent(dataMatrix, classLabels, numIter=500):
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha会随着迭代次数不断减少，但存在常数项，它不会小到0
            # 这种设置可以缓解数据波动
            alpha = 4/(1.0+j+i) + 0.0001
            # 通过随机选取样本来更新回归系数
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

'''
画出数据集和logistic回归最佳拟合直线的函数
'''
def plotBestFit(dataMat, labelMat, weights):
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    # 根据类别分别保存点
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    # plt.subplot(2,3,1)表示把图标分割成2*3的网格，也可以简写plt.subplot(231)
    # 第一个参数是行数，第二个参数是列数，第三个参数表示图形的标号
    ax = fig.add_subplot(111)
    # scatter 散列点
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 此处设置了sigmoid的z为0， 因为0是两个分类的分界处
    # 即：0=w0x0+w1x1+w2x2
    # 注意：x0=1, x1=x, 解出x2=y
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x, y.transpose())
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    
    
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(dataMat, labelMat, weights)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    