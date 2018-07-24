# coding: utf-8
from math import log
import operator

'''
创建数据集
dataSet    数据集，包含样本class
labels    特征名称标签
'''
def createDataSet():    
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
#     labels = ['no surfacing', 'no surfacing', 'no surfacing', 'flippers', 'flippers']
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

'''
计算香农熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 统计每个类别出现的次数，保存在字典labelCounts中
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # 如果当前键值不存在，则扩展字典并将当前键值加入字典
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    
    return shannonEnt


'''
选择最好的数据集划分方式
输入：数据集
输出：最优分类的特征的index
'''
def chooseBestFeatureToSplit(dataSet):
    # 计算特征数量
    numFeature = len(dataSet[0]) - 1
    
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    
    for i in range(numFeature):
        # 创建唯一的分类标签列表
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        # 计算每种划分方式的信息熵
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 计算最好的信息增益，即infoGain越大划分效果越好
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
按照给定特征划分数据集。以dataSet的第axis个特征的value值划分dataSet
dataSet    待划分的数据集
axis        划分数据集的第axis特征
value        特征的返回值（比较值）
'''
def splitDataSet(dataSet, axis, value):
    
    retDataSet = []
    
    # 遍历数据集中的每个元素，一旦发现符合要求的值，则将其添加到新创建的列表中
    for featureVec in dataSet:
        if featureVec[axis] == value:
            reducedFeatureVec = featureVec[:axis]
            reducedFeatureVec.extend(featureVec[axis+1:])
            retDataSet.append(reducedFeatureVec)
            # a=[1,2,3] b=[4,5,6]
            # a.append(b)=[1,2,3,[4,5,6]]
            # a.extend(b)=[1,2,3,4,5,6]
    return retDataSet



'''
投票表决函数
输入：classList，标签集合，本例为['yes','yes','no','no','no']
输出：得票数最多的分类名称
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    # 把分类结果进行排序，然后返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
创建树
输入：数据集和标签列表
输出：树的所有信息
'''
def createTree(dataSet, labels):
    # classList为数据集的所有类标签
    classList = [example[-1] for example in dataSet]
#     print('classList=', classList)
    
    # 停止条件1：所有类标签完全相同，直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    # 停止条件2：遍历完所有特征时仍不能将数据集划分成仅包含唯一类别的分组，则返回出现次数最多的类标签
    # 即只有一个特征的时候，返回出现次数最多的类标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    # 选择最优分类特征
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeature] # labels是特征名？？
#     print('--bestFeature=', bestFeature)
#     print('--bestFeatLabel=', bestFeatLabel)
    
    # myTree存储树的所有信息
    myTree = {bestFeatLabel:{}}
    
    # 以下得到列表包含的所有属性值
    del(labels[bestFeature])
    featureValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featureValues)
    
    # 遍历当前选择特征包含的所有属性值
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

'''
决策树的分类函数
inputTree    训练好的树信息
featLabels    标签列表
testVec        测试向量
'''
def classify(inputTree, featureLabels, testVec):
    print('*=*=*=*=*=*=*')
    print('featureLabels = ', featureLabels)
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]
    secondDict = inputTree[firstStr]
    print('firstSides = ', firstSides)
    print('firstStr = ', firstStr)
    print('secondDict = ', secondDict)
    
    # 将标签字符串转换成索引
    featureIndex = featureLabels.index(firstStr)
    print('featureIndex = ', featureIndex)
    key = testVec[featureIndex]
    print('key = ', key)
    valueOfFeature = secondDict[featureIndex]
    print('valueOfFeature = ', valueOfFeature)
    # 递归遍历整棵树，比较testVec变量中的值与树节点的值，如果到达叶子节点，则返回当前节点的分类标签
    if isinstance(valueOfFeature, dict):
        classLabel = classify(valueOfFeature, featureLabels, testVec)
    else: classLabel = valueOfFeature
    return classLabel
        
'''
使用pickle模块存储决策树
'''
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb+')
    pickle.dump(inputTree, fw)
    fw.close()
    
'''
导入决策树模块
'''
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)



if __name__ == '__main__':
    dataSet, labels = createDataSet()
#     shannonEnt = calcShannonEnt(dataSet)
#     print(shannonEnt)
    print('labels = ', labels)
    featureLabels = labels[:]
    myTree = createTree(dataSet, labels)
    print('labels = ', labels)
    print('featureLabels = ', featureLabels)
    ans = classify(myTree, featureLabels, [1,0])
    print('ans = ', ans)
    
    # 存取操作
    storeTree(myTree, './mt.txt')
    myTree2 = grabTree('./mt.txt')
    print(myTree2)
    
    
    
    