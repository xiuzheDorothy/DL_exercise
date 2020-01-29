def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()     # strip去换行符
        listFromLine = line.split('\t')     # split 去掉\t
        returnMat[index, :] = listFromLine[0:3]     # returnMax 每一行为array结构，可以使用索引，不太明白为什么直接生成array
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
"""
用来实现数据的导入，源数据格式如下
40920	8.326976	0.953952	3
1000行4列，中间用'\t'隔开，最后一列为训练集的标签
所以：前三列用returnMax存储，最后一列用classLabelVector存储
"""