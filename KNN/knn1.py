# coding: utf-8
from numpy import *
import operator

class KNN:
    def createDataSet(self):
        group = array([[1.0, 1.1],
                       [1.0, 1.0], 
                       [0, 0],
                       [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        return group, labels
    
    '''
    inX    用于分类的输入向量
    dataSet    训练样本集
    labels    标签向量
    k        用于选择最近邻居的数目
    '''
    def classify0(self, inX, dataSet, labels, k):
        # shape返回矩阵的[行数,列数]，shape[0]获取数据集的行数，即样本的数量
        dataSetSize = dataSet.shape[0]
        # 下面的求距离的过程是按照欧式距离的公式计算的，根号(x^2+y^2)
        # ！！！tile
        diffMat = tile(inX, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat ** 2
        # axis=1表示横轴，即按照横轴sum
        sqDistances = sqDiffMat.sum(axis=1)
        # 对平方和开根号
        distances = sqDistances ** 0.5
        # 按照升序进行快速排序，返回的是原数组的下标
        sortedDistIndicies = distances.argsort()
        classCount = {}
        # 投票过程，统计前k个最近的样本所属类别包含的样本个数
        for i in range(k):
            voteIlable = labels[sortedDistIndicies[i]]
            classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    '''
            归一化
    '''
    def autoNorm(self, dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minVals, (m,1))
        normDataSet = normDataSet / tile(ranges, (m,1))
        return normDataSet, ranges, minVals


if __name__ == '__main__':      
    knn = KNN() 
    group, lables = knn.createDataSet()
#     print(group)
#     print(lables)
    print(knn.classify0([0,0], group, lables, 3))