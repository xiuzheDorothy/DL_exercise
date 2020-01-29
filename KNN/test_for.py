from __future__ import print_function # __future__模块可以引用未来版本python库中的函数，其实我的版本已经是3.x，所以没有必要走这一步
from numpy import *
import operator
from os import listdir
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

fr=open("E:\MachineLearning\KNN\dataset\datingTestSet2.txt")
numberOfLines=len(fr.readlines())
fr.seek(0)
returnMat=zeros((numberOfLines,3))
classLabelVector=[]
index=0
for line in fr.readlines():
    line=line.strip()
    listFromLine=line.split('\t')
    returnMat[index,:]=listFromLine[0:3]
    classLabelVector.append(int(listFromLine[-1]))  # classLabelVector 与 returnMat 一一对应，前者为所属标签，后者为具体数据
    index+=1

labels=classLabelVector

plt.figure(figsize=(8, 5), dpi=80)
axes = plt.subplot(111)
# 将三类数据分别取出来
# x轴代表飞行的里程数
# y轴代表玩视频游戏的百分比
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []

for i in range(len(labels)):
    if labels[i] == 1:  # 不喜欢
        type1_x.append(returnMat[i][0])
        type1_y.append(returnMat[i][1])

    if labels[i] == 2:  # 魅力一般
        type2_x.append(returnMat[i][0])
        type2_y.append(returnMat[i][1])

    if labels[i] == 3:  # 极具魅力
        type3_x.append(returnMat[i][0])
        type3_y.append(returnMat[i][1])

type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')

plt.xlabel('Miles earned per year')
plt.ylabel('Percentage of events spent playing video games')
axes.legend((type1, type2, type3), (u'dislike', u'Charismatic', u'Very attractive'), loc=2,)

plt.show()

"""
下面进行归一化操作
公式：线性转换：  newValue=(oldValue-min)/(max-min)  max和min为该组数据集中的最大最小特征值
      对数转换：  y=log10(x)
      反余切转换：y=arctan(x)*2/PI　
"""

normMat=zeros((numberOfLines,3))
for i in [0,1,2]:
    max=returnMat[:,i].max()
    min=returnMat[:,i].min()
    for j in range(1000):
        normMat[j,i]=(returnMat[j,i]-min)/(max-min)
# done

"""
对于每一个在数据集中的数据点：
    计算目标的数据点（需要分类的数据点）与该数据点的距离
    将距离排序：从小到大
    选取前K个最短距离
    选取这K个中最多的分类类别
    返回该类别来作为目标数据点的预测值
"""

inx=[0.40166314, 0.56719748, 0.52034602] # 作为输入的样本
dataSetSize=normMat.shape[0]
diffMat = tile(inx, (dataSetSize,1))-normMat
sqDiffMat = diffMat**2
distances=(sqDiffMat.sum(axis=1))**0.5
sortedDistIndicies = distances.argsort() # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引号)，所以并不会影响对应的label的值
# k取5
classCount={}
for i in range(5):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
