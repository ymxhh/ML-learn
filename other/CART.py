# '''
# Created on 2018��3��21��
# 
# @author: 27419
# '''
# from numpy import *
# # from numpy import shape
# '''
# CART
# �㷨���̣�
# ��1��������������������������������һ���ݹ麯����
# �ú�������Ҫ�����ǰ���CART�Ĺ����������������ĸ�����֧�ڵ㣬��������ֹ���������㷨��
# a)������Ҫ��������ݼ�������ǩ
# b)ʹ����Сʣ�෽���ж��ع��������Ż��֣������������Ļ��ֽڵ�--��Сʣ�෽���Ӻ���
# c)�ڻ��ֽڵ㻮�����ݼ�Ϊ������--�������ݼ��Ӻ���
# d)���ݶ������ݵĽ���������µ����ҽڵ㣬��Ϊ��������������֧
# e)�����Ƿ���ϵݹ����ֹ����
# f)�����ֵ��½ڵ���������ݼ�������ǩ��Ϊ���룬�ݹ�ִ����������
# ��2��ʹ����Сʣ�෽���Ӻ������������ݼ����е����л��ַ�������С�����ֵ��
# ��3���������ݼ������ݸ����ķָ��кͷָ�ֵ�����ݼ�һ��Ϊ�����ֱ𷵻ء�
# ��4����֦���ԣ�ʹ���ȼ�֦�ͺ��֦���ԶԼ�����ľ��������м�֦
# '''
# 
# '''
# ��Ԫ�з����ݼ�
# dataSet����������ݼ�
# feature��������
# value�����ֵ��ȡֵ
# '''
# def binSplit(dataSet, feature, value):
#     mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
#     mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
#     return mat0, mat1
# 
# '''
# ѡ�����ŷָ��
# leafType��Ҷ�ӽڵ����Իع麯��
# errType����Сʣ�෽��ʵ�ֺ���
# ops������ķ����½�ֵ����С�з�������
# '''
# def getBestFeat(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
#     tolS = ops[0]   # ����ķ����½�ֵ
#     tolN = ops[1]   # ��С�з�������
#     
#     #---�㷨��ֹ����1��ʼ---
#     splitdataSet = set(dataSet[:,-1].T.tolist()[0])
#     if len(splitdataSet) == 1:
#         return None, leafType(dataSet)
#     
#     #---����dataSet���е����Ż��ַ�������С�����ֵ---
#     m,n = shape(dataSet)    # �������ݼ�������������
#     S = errType(dataSet)    # �����������ݼ��Ļع鷽��S
#     
#     # ��ʼ�����Ų�������󷽲���Ż����С����Ż���ֵ
#     bestS = inf; bestIndex=0; bestValue=0
#     for featIndex in range(n-1):    # ����ѭ��
#         for splitVal in set(dataSet[:,featIndex]):  # ����ѭ��--ȥ��
#             mat0, mat1 = binSplit(dataSet, featIndex, splitVal)     # ��Ԫ�������ݼ�
#             # mat0������С��tolN����mat1������С��tolN
#             if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
#                 continue
#             newS = errType(mat0) + errType(mat1)    # ������С�����
#             if newS < bestS:
#                 bestIndex = featIndex   # �������� <- ��������
#                 bestValue = splitVal    # ����ֵ  <- �ָ�ֵ
#                 bestS = newS            # bestS <- newS
#     #---DataSet�����Ż��ֲ�������������С�����ֵ�������
#     
#     #---�㷨��ֹ����2��ʼ�����ص���ֵ�ڵ�����
#     if (S-bestS) < tolS:
#         return None, leafType(dataSet)
#     
#     #---�㷨��ֹ����3��ʼ
#     # ��Ԫ�������ݼ����������кͻ���ֵ�ָ�dataSet
#     mat0, mat1 = binSplit(dataSet, bestIndex, bestValue)
#     # mat0������С��tolN��mat1������С��tolN
#     if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
#         return None, leafType(dataSet)
#     # �㷨��ֹ��ǰ3�������Ļ�����ΪNone��˵��ΪҶ�ӽڵ㣬��֧���������ֽ���
#     #---�㷨��ֹ����4��ʼ�����ص��������ڵ�����
#     # �������������Ļ����кͻ���ֵ�����ع�������ݹ黮��
#     return bestIndex, bestValue