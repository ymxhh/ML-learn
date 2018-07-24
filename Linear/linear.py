# coding:utf-8
from numpy import *

'''
线性回归     & 局部加权线性回归
'''


'''
导入数据
'''
def loadDataSet(filename):
    numFeature = len(open(filename).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeature):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

'''
求回归函数
'''
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:#判断行列式是否为0
        print("This matrix is singular, cannot do inverse")
        return
    # .I用作求矩阵的逆矩阵
    ws = xTx.I * (xMat.T*yMat)#也可以用NumPy库的函数求解：ws=linalg.solve(xTx,xMat.T*yMatT)
    return ws


'''
局部加权线性回归函数
k值控制衰减速度，且k值越小被选用于训练回归模型的数据集越小
'''
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T    
    m = shape(xMat)[0]  # 行
    # 初始化weights
    weights = mat(eye((m))) #创建对角矩阵
    for j in range(m):        
        diffMat = testPoint - xMat[j,:]
        #高斯核计算权重
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr,xArr,yArr,k=1.0):
    '''为数据集中每个点调用lwlr()'''
    m = shape(testArr)[0]   # 行
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

if __name__ == '__main__':
    '''局部加权线性回归'''
    xArr, yArr = loadDataSet('ex0.txt')
    # 拟合
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    # 绘图
    xMat = mat(xArr)
    yMat = mat(yArr)
    strInd = xMat[:,1].argsort(0)
    xSort = xMat[strInd][:,0,:]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[strInd])
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0], s=2, c='red')
    plt.show()
    
    '''线性回归'''
    '''
    xArr, yArr = loadDataSet('./ex0.txt')
#     print('xArr:')
#     for line in xArr:
#         print(line)
#     print('yArr:')
#     print(yArr)
    
    ws = standRegres(xArr, yArr)
#     print('ws:', ws)
    xMat = mat(xArr)
    yMat = mat(yArr)
    # 预测值
    yHat = xMat * ws
#     print('yHat=', yHat)
      
    # 计算预测值和真实值的相关性
    corrcoef(yHat.T, yMat)
     
    # 绘制数据集散点图和最佳拟合直线图
    # 创建图像并绘出原始的数据
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # a是个矩阵或者数组，a.flatten()就是把a降到一维，默认是按横的方向降
    # a.flatten('F')按竖的方向降
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
      
    # 绘最佳拟合直线，需先要将点按照升序排列
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1], yHat)
    plt.show()
    '''
    
    
    
    
    
    
    
    
    
    