def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 距离度量 度量公式为欧氏距离
    diffMat = tile(inX, (dataSetSize, 1)) – dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # 将距离排序：从小到大
    sortedDistIndicies = distances.argsort()
    # 选取前K个最短距离， 选取这K个中最多的分类类别
    classCount = {}
    for i in range(k)：
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1


sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
return sortedClassCount[0][0]