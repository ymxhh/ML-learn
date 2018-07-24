# coding: utf-8

from numpy import *

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        floatLine = map(float, curLine)
        dataMat.append(floatLine)
    return dataMat

def binSplitDataSet(dataSet, featureIndex, splitValue):
    mat0 = dataSet[nonzero(dataSet[:, featureIndex]> splitValue)[0],:]
    mat1 = dataSet[nonzero(dataSet[:, featureIndex] <= splitValue)[0],:]
    return mat0, mat1


'''负责生成叶节点'''
def regLeaf(dataSet):
    # 在chooseBestSplit()函数确定不再对数据进行划分时，将调用本函数来得到叶节点的模型
    # 在回归树中，该模型其实就是目标变量的均值
    return mean(dataSet[:,-1].tolist())


'''
误差估计函数，该函数在给定的数据上计算目标变量的平方误差，这里直接调用均方差函数
'''
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]   # 返回总方差

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    featureIndex, splitValue = chooseBestSplit(dataSet, leafType, errType, ops)
    if featureIndex == None: return splitValue
    retTree = {}
    retTree['spInd'] = featureIndex
    retTree['spVal'] = splitValue
    # 将数据集分为两份，之后递归调用继续划分
    lSet, rSet = binSplitDataSet(dataSet, featureIndex, splitValue)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    
    return retTree



def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # ops为用户指定参数，用于控制函数的停止时机
    tolS = ops[0]; tolN = ops[1]
    # 如果所有值相等则退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    # 在所有可能的特征及其可能取值上遍历，找到最佳的切分方式
    # 最佳切分也就是使得切分后能达到最低误差的切分
    for featureIndex in range(n-1):
        for splitValue in set(dataSet[:, featureIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featureIndex, splitValue)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featureIndex
                bestValue = splitValue
                bestS = newS
    # 如果误差减小不大则退出
    if (S-bestS) < tolS:
        return None, leafType(dataSet)
    
    # 如果切分出的数据集很小则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    
    # 提前终止条件都不满足，返回切分特征和特征值
    return bestIndex, bestValue



if __name__ == '__main__':
    data = loadDataSet('./ex00-3.txt')
    dataSet = mat(data)
    createTree(dataSet)













