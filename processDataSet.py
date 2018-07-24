# coding: utf-8
import os

# 去掉数据集中的空格

'''
去掉abalone.txt中空格行
'''
def noBlankSpace(filenameIn, filenameOut):
    fr = open(filenameIn)
    
#     if os.path.exists(filenameOut):
#         print('file exist')
    
    fw = open(filenameOut, 'a+') # 按行追加
    count = 0
    for line in fr.readlines():
        count += 1
        if count % 2 != 0:
            fw.write(line)
    fw.close()
    fr.close()
    
if __name__ == '__main__':
    filenameIn = 'D:/eclipse-workspace2/ML-learn/DecisionTree/ex00.txt'
    filenameOut = 'D:/eclipse-workspace2/ML-learn/DecisionTree/ex00-3.txt'
#     filenameIn = '../DecisionTree/ex00.txt'
#     filenameOut = '../DecisionTree/ex00-2.txt'
    noBlankSpace(filenameIn, filenameOut)