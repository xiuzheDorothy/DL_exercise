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
parse_cfg('./cfg/yolov3.cfg')