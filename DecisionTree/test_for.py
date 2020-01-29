from __future__ import print_function
print(__doc__)
import operator
from math import log
from collections import Counter
#import decisionTreePlot as dtPlot 这一步是已经建好决策树后的调包使用，这里先不予考虑

#dataset
dataset=[[1,1,'yes'],
         [1,1,'yes'],
         [1, 0, 'no'],
         [0, 1, 'no'],
         [0, 1, 'no']]
labels=['no surfacing','flippers']

"""
calculate the entropy 计算香农熵
"""

numEntries=len(dataset)
labelCounts={}
# the number of unique elements and their occurrence
for featVec in dataset:
    currentlabel=featVec[-1]
    if currentlabel not in labelCounts.keys():
        labelCounts[currentlabel]=0
    labelCounts[currentlabel]+=1    # each key/value records the amount of the current category appears
shannonEnt=0.0
for key in labelCounts:
    prob=float(labelCounts[key])/numEntries # 频率->概率
    shannonEnt=-(prob*log(prob,2))

# 按照给定特征划分数据集
#def splitDataSet(dataSet, index, value):
"""
 Args:
        dataSet 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征（特征列1，特征列2）
        value 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
        
在这里先测试第一列所代表的特征，将值为value的划分到一个子数据集，value这里设置为1
"""
index=0
value=1
featVec=[]
retDataSet=[]
for featVec in dataset:
    if featVec[index]==value:
        reducedFeatVec=featVec[:index]
        reducedFeatVec.extend(featVec[index + 1:]) # 去掉了index这一列的数据
        retDataSet.append(reducedFeatVec)
        # 有点类似于通过遍历得到的序列还原为二叉树的过程
        # retDataSet为最后得到的子树


