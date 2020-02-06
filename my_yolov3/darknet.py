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
    prev_filters = 3
    output_filters = []# 我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。

    for index , x in enumerate(blocks[1:]):
        module=nn.Sequential()

        if (x['type'])=='convolutional':
            activation=x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False #卷积层后接BN就不需要bias
            except:
                batch_normalize = 0
                bias = True #卷积层后无BN层就需要bias

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding: # config文件中的pad表示这一层是否经过padding操作（数值只为1）
                        # 因而pad的值要如下实际计算
                pad = (kernel_size-1)//2    # 向下除接近的整数
            else:
                pad = 0

            # 下来开始创建层：
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias = bias)
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

class Darknet((nn.Module)):
    def __init__(self, cgffile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = creat_modules(self.blocks)

    def forward(self, x, CUDA):# x为输入
        modules = self.blocks[1:]  # 除了net块之外的所有，forward这里用的是blocks列表中的各个block块字典
        outputs = {}  # We cache the outputs for the route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                # 如果只有一层时。从前面的if (layers[0]) > 0:语句中可知，如果layer[0]>0，则输出的就是当前layer[0]这一层的特征,如果layer[0]<0，输出就是从route层(第i层)向后退layer[0]层那一层得到的特征
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                # 第二个元素同理
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)  # 第二个参数设为 1,这是因为我们希望将特征图沿anchor数量的维度级联起来。


            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]  # 求和运算，它只是将前一层的特征图添加到后面的层上而已

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # 从net_info(实际就是blocks[0]，即[net])中get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data  # 这里得到的是预测的yolo层feature map
                # 在util.py中的predict_transform()函数利用x(是传入yolo层的feature map)，得到每个格子所对应的anchor最终得到的目标
                # 坐标与宽高，以及出现目标的得分与每种类别的得分。经过predict_transform变换后的x的维度是(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if not write:  # if no collector has been intialised. 因为一个空的tensor无法与一个有数据的tensor进行concatenate操作，
                    detections = x  # 所以detections的初始化在有预测值出来时才进行，
                    write = 1  # 用write = 1标记，当后面的分数出来后，直接concatenate操作即可。
                else:
                    detections = torch.cat((detections, x), 1)  # 将在3个不同level的feature map上检测结果存储在 detections 里

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)  # 这里读取first 5 values权重
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)  # 加载 np.ndarray 中的剩余权重，权重是以float32类型存储的

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]  # blocks中的第一个元素是网络参数和图像的描述，所以从blocks[1]开始读入

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])  # 当有bn层时，"batch_normalize"对应值为1
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model 将从weights文件中得到的权重bn_biases复制到model中(bn.bias.data)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:  # 如果 batch_normalize 的检查结果不是 True，只需要加载卷积层的偏置项
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)