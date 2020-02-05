import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# from util import *

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0,
                                      1))  # img是【h,w,channel】，这里的img[:,:,::-1]是将第三个维度channel从opencv的BGR转化为pytorch的RGB，然后transpose((2,0,1))的意思是将[height,width,channel]->[channel,height,width]
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    输入: 配置文件路径
    返回值: 列表对象,其中每一个元素为一个字典类型对应于一个要建立的神经网络模块（层）

    """
    # 加载文件并过滤掉文本中多余内容
    file = open(cfgfile, 'r')
    file.seek(0)
    lines = file.read().split('\n')  # store the lines in a list等价于readlines

    lines = [x for x in lines if len(x) > 0]  # 去掉空行
    lines = [x for x in lines if x[0] != '#']  # 去掉以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)
    # 以上这些操作叫做列表表达式
    # cfg文件中的每个块用[]括起来最后组成一个列表。一个block存储一个块的内容，即每个层用一个字典block存储。
    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # 这是cfg文件中一个层(块)的开始
            if len(block) != 0:  # 如果块内已经存了信息, 说明是上一个块的信息还没有保存
                blocks.append(block)  # 那么这个块（字典）加入到blocks列表中去
                block = {}  # 覆盖掉已存储的block,新建一个空白块存储描述下一个块的信息(block是字典)
            block["type"] = line[1:-1].rstrip()  # 把cfg的[]中的块名作为键type的值
        else:
            key, value = line.split("=")  # 按等号分割
            block[key.rstrip()] = value.lstrip()  # 左边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对
    blocks.append(block)  # 退出循环，将最后一个未加入的block加进去
    # print('\n\n'.join([repr(x) for x in blocks]))
    return blocks

# parse_cfg('./cfg/yolov3.cfg')
# print(0)

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def creat_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_fliters = 3
    output_filters = []# 我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。

    for index , x in enumerate(blocks):
        module=nn.Sequential()

        if (x['type'])=='convolutional':
            activation=x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False #卷积层后接BN就不需要bias
            except:
                batch_normalize = 0
                bias = True #卷积层后无BN层就需要bias

            filters = int(x['filters'])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding: # config文件中的pad表示这一层是否经过padding操作（数值只为1）
                        # 因而pad的值要如下实际计算
                pad = (kernel_size-1)//2    # 向下除接近的整数
            else:
                pad = 0

            # 下来开始创建层：
            conv = nn.Conv2d(prev_fliters,filters,kernel_size,stride,pad,bias = bias)
            module.add_module('conv_{0}'.format(index) ,conv) #format 格式化函数

            #batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            #activation
            if  activation=='leaky':
                active_layer = nn.LeakyReLU(0.1,inplace=True) #LeakyReLU函数在负数范围的斜率为0.1
                module.add_module("leaky_{0}".format(index), active_layer)
        elif (x['type'] == 'upsample'):
            stride = int(x['stride'])
            # #这个stride在cfg中就是2，所以下面的scale_factor写2或者stride是一样的
            upsample = nn.Upsample(scale_factor=2,mode='nearnst')
            module.add_module("upsample_{}".format(index), upsample)

        # route层
        # 当layer取值为正时，输出这个正数对应的层的特征
        # 如果layer取值为负数，输出route层向后退layer层对应层的特征
        elif (x['type'] == 'route'):
            x['layers'] = x['layers'].split(',')
            start=int(x['layers'][0])
            try:
                end=int(x['layers'][0])
            except:
                end=0
            if start>0:
                start=start-index
            if end > 0:  # 若end>0，由于end= end - index，再执行index + end输出的还是第end层的特征
                end = end - index
            route=EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:  # 若end<0，则end还是end，输出index+end(而end<0)故index向后退end层的特征。
                filters = output_filters[index + start] + output_filters[index + end]
            else:  # 如果没有第二个参数，end=0，则对应下面的公式，此时若start>0，由于start = start - index，再执行index + start输出的还是第start层的特征;若start<0，则start还是start，输出index+start(而start<0)故index向后退start层的特征。
                filters = output_filters[index + start]
                # shortcut corresponds to skip connection
        # shortcut
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()  # 使用空的层，因为它还要执行一个非常简单的操作（加）。没必要更新 filters 变量,因为它只是将前一层的特征图添加到后面的层上而已。
            module.add_module("shortcut_{}".format(index), shortcut)

        ##yolo
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)# 锚点,检测,位置回归,分类，这个类见predict_transform中
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)# 最后将打包好的module放入module_list中
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class Darknet((nn.Module):
