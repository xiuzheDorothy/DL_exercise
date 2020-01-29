dataSet=[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
C1=[] # C1中存放所有的出现过的项
for transaction in dataSet:
    for item in transaction:
        if not [item] in C1:
            C1.append([item])
C1.sort()

'''
>>>def square(x) :            # 计算平方数
...     return x ** 2
... 
>>> map(square, [1,2,3,4,5])   # 计算列表各个元素的平方
[1, 4, 9, 16, 25]
>>> map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # 使用 lambda 匿名函数
[1, 4, 9, 16, 25]
 
# 提供了两个列表，对相同位置的列表数据进行相加
>>> map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
[3, 7, 11, 15, 19]
'''

map(frozenset, C1) #???不懂
#frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素
'''
Python 2 map 返回 list object
Python 3 map 返回 map object 且只能使用一次
'''

# 下面计算支持度
ssCnt = {}