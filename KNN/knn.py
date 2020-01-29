from __future__ import print_function

# from numpy import * 等同于 import numpy
# 以掉用numpy中的random模块为例，import numpy要用numpy.random，from numpy import *只用random即可
from numpy import *
import operator     # 运算符模块operator
from os import listdir
from collections import Counter

def createDataSet():
    """
    创建数据集和标签
     调用方式
     import kNN
     group, labels = kNN.createDataSet()
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k)

def file2matrix(filename):
    fr=open(filename)
    number_of_lines=len(fr.readlines())
    returnMat=zeros(number_of_lines,3)
    classLableVector=[] # prepare matrix to return
    fr.seek(0)
    index=0     # ?
    for line in fr.readlines()
        

