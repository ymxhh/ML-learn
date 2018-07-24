# '''
# Created on 2018年3月21日
# 
# @author: 27419
# '''
# from numpy import *
# # from numpy import shape
# '''
# CART
# 算法流程：
# （1）决策树主函数：决策树的主函数是一个递归函数。
# 该函数的主要功能是按照CART的规则生长出决策树的各个分支节点，并根据终止条件结束算法。
# a)输入需要分类的数据集和类别标签
# b)使用最小剩余方差判定回归树的最优划分，并创建特征的划分节点--最小剩余方差子函数
# c)在划分节点划分数据集为两部分--二分数据集子函数
# d)根据二分数据的结果构建出新的左右节点，作为树生长的两个分支
# e)检验是否符合递归的终止条件
# f)将划分的新节点包含的数据集和类别标签作为输入，递归执行上述步骤
# （2）使用最小剩余方差子函数，计算数据集各列的最有划分方差、划分列、划分值。
# （3）二分数据集，根据给定的分割列和分割值将数据集一分为二，分别返回。
# （4）剪枝策略：使用先剪枝和后剪枝策略对计算出的决策树进行剪枝
# '''
# 
# '''
# 二元切分数据集
# dataSet：输入的数据集
# feature：特征列
# value：二分点的取值
# '''
# def binSplit(dataSet, feature, value):
#     mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
#     mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
#     return mat0, mat1
# 
# '''
# 选择最优分割点
# leafType：叶子节点线性回归函数
# errType：最小剩余方差实现函数
# ops：允许的方差下降值，最小切分样本数
# '''
# def getBestFeat(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
#     tolS = ops[0]   # 允许的方差下降值
#     tolN = ops[1]   # 最小切分样本数
#     
#     #---算法终止条件1开始---
#     splitdataSet = set(dataSet[:,-1].T.tolist()[0])
#     if len(splitdataSet) == 1:
#         return None, leafType(dataSet)
#     
#     #---计算dataSet各列的最优划分方差、划分列、划分值---
#     m,n = shape(dataSet)    # 返回数据集的行数和列数
#     S = errType(dataSet)    # 计算整个数据集的回归方差S
#     
#     # 初始化最优参数：最大方差、最优划分列、最优划分值
#     bestS = inf; bestIndex=0; bestValue=0
#     for featIndex in range(n-1):    # 按列循环
#         for splitVal in set(dataSet[:,featIndex]):  # 按行循环--去重
#             mat0, mat1 = binSplit(dataSet, featIndex, splitVal)     # 二元划分数据集
#             # mat0的行数小于tolN，或mat1的行数小于tolN
#             if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
#                 continue
#             newS = errType(mat0) + errType(mat1)    # 计算最小方差和
#             if newS < bestS:
#                 bestIndex = featIndex   # 最优索引 <- 特征索引
#                 bestValue = splitVal    # 最优值  <- 分隔值
#                 bestS = newS            # bestS <- newS
#     #---DataSet的最优划分参数：方差、划分列、划分值计算结束
#     
#     #---算法终止条件2开始：返回的是值节点类型
#     if (S-bestS) < tolS:
#         return None, leafType(dataSet)
#     
#     #---算法终止条件3开始
#     # 二元划分数据集：按划分列和划分值分隔dataSet
#     mat0, mat1 = binSplit(dataSet, bestIndex, bestValue)
#     # mat0的行数小于tolN或mat1的行数小于tolN
#     if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
#         return None, leafType(dataSet)
#     # 算法终止的前3个条件的划分列为None，说明为叶子节点，本支分类树划分结束
#     #---算法终止条件4开始：返回的是子树节点类型
#     # 返回最优特征的划分列和划分值，但回归树还需递归划分
#     return bestIndex, bestValue