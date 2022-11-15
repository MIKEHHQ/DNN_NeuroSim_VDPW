'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d, QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
from torchvision.models.utils import load_state_dict_from_url
name=0

def Depthwise(in_planes, stride=1, groups=3, dilation=1,padding=1, args=None, logger=None):
    """3x3 convolution with padding"""
    global name
    if args.mode == "WAGE":
        conv2d = QConv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=padding,groups=in_planes, # padding=dilation -->padding=padding
                         logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='Depthwise' + '_' + str(name) + '_', model=args.model)

    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, in_planes, kernel_size=3, stride=stride,groups=in_planes,
                         padding=padding, bias=False, dilation=dilation,
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='Depthwise' + '_' + str(name) + '_')
    name += 1
    return conv2d


def Pointwise(in_planes, out_planes, stride=1,padding=0, args=None, logger=None):
    """1x1 convolution"""
    global name
    if args.mode == "WAGE":
        conv2d = QConv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=padding, logger=logger, wl_input=args.wl_activate,
                         wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='Pointwise' + '_' + str(name) + '_', model=args.model)
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=padding, bias=False,
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='Pointwise' + '_' + str(name) + '_')
    name += 1
    return conv2d


def Conv2d(in_planes, out_planes, kernel_size, stride, padding,groups, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":
        conv2d = QConv2d(in_planes, out_planes, kernel_size, stride, padding,groups=groups, logger=logger, wl_input=args.wl_activate,
                         wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='Conv' + '_' + str(name) + '_', model=args.model)
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False,groups=groups,
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='Conv' + '_' + str(name) + '_')
    name += 1
    return conv2d


def Linear(in_planes, out_planes, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":
        linear = QLinear(in_planes, out_planes,
                         logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                         wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                         cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='FC' + '_' + str(name) + '_', model=args.model)
    elif args.mode == "FP":
        linear = FLinear(in_planes, out_planes, bias=False,  # Flase to True to False
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='FC' + '_' + str(name) + '_')
    name += 1
    return linear





class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1,args=None, logger=None):
        super(Block, self).__init__()
        self.conv1 = Depthwise(in_planes,stride=stride, padding=1, groups=in_planes, args=args, logger=logger)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = Pointwise(in_planes, out_planes,stride=1,padding=0, args=args, logger=logger)
        self.bn2 = nn.BatchNorm2d(out_planes)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class FuSeHalfBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1,args=None, logger=None):
        super(FuSeHalfBlock, self).__init__()
        self.conv1_h = Conv2d(in_planes//2, in_planes//2, kernel_size=(1,3), stride=stride, padding=(0,1), groups=in_planes//2, args=args, logger=logger)
        self.bn1_h = nn.BatchNorm2d(in_planes//2)
        self.conv1_v = Conv2d(in_planes//2, in_planes//2, kernel_size=(3,1), stride=stride, padding=(1,0), groups=in_planes//2, args=args, logger=logger)
        self.bn1_v = nn.BatchNorm2d(in_planes//2)
        self.conv2 = Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0,groups=1, args=args, logger=logger)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out1, out2 = x.chunk(2,1)
        out1 = self.bn1_h(self.conv1_h(out1))
        out2 = self.bn1_v(self.conv1_v(out2))
        out  = torch.cat([out1, out2], 1)
        # out = F.relu(out)
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class FuSeFullBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1,args=None, logger=None):
        super(FuSeFullBlock, self).__init__()
        self.conv1_h = Conv2d(in_planes, in_planes, kernel_size=(1,3), stride=stride, padding=(0,1), groups=in_planes, args=args, logger=logger)
        self.bn1_h = nn.BatchNorm2d(in_planes)
        self.conv1_v = Conv2d(in_planes, in_planes, kernel_size=(3,1), stride=stride, padding=(1,0), groups=in_planes, args=args, logger=logger)
        self.bn1_v = nn.BatchNorm2d(in_planes)
        self.conv2 = Conv2d(2*in_planes, out_planes, kernel_size=1, stride=1, padding=0,groups=1, args=args, logger=logger)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out1 = self.bn1_h(self.conv1_h(x))
        out2 = self.bn1_v(self.conv1_v(x))
        out  = torch.cat([out1, out2], 1)
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    # cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    cfg = [64]

    def __init__(self,block, num_classes, args, logger):
        super(MobileNet, self).__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1,groups=1,args=args, logger=logger) # change stride from 1 to 2
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(block, in_planes=32, args=args, logger=logger)
        self.conv2 = Conv2d(64, 32, 3, stride=1, padding=1, groups=1, args=args, logger=logger)
        # self.conv2 = Conv2d(1024,256,3, stride=1,padding=1, groups=1, args=args, logger=logger)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.linear = Linear(32, num_classes, args=args, logger=logger)

    def _make_layers(self,block, in_planes, args, logger):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(block(in_planes, out_planes, stride,args=args,logger=logger))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print('^^^^^^^^^^^before the  layers mbv1 is ', out.shape)
        out = self.layers(out)
        out = self.conv2(out)
        # print('######the layer shape are',out.shape)
        # print('**********************before the  avg pool2d is ', out.shape)
        # out = F.avg_pool2d(out, 2)
        out = self.avgpool(out)

        # print('avg pool is ',self.avgpool(out).shape)
        # print('____________________________after avg pool2d is ',out.shape)
        out = out.view(out.size(0), -1)
        # print('before the linear is ',out.shape)
        out = self.linear(out)
        # print('after the linear is ', out.shape)
        return out

def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

def MobileNetV1(num_classes=10, args=None, logger=None, pretrained=None):
    model = MobileNet(Block,num_classes, args=args, logger=logger)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    return model

def MobileNetV1FuSeHalf(num_classes=10, args=None, logger=None, pretrained=None):
    model = MobileNet(FuSeHalfBlock,num_classes, args=args, logger=logger)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    return model

def MobileNetV1FuSeFull(num_classes=10, args=None, logger=None, pretrained=None):
    model = MobileNet(FuSeFullBlock,num_classes, args=args, logger=logger)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    return model
# def MobileNetV1(num_classes=10, args=None, logger=None, pretrained=None):
#     model = MobileNet(Block,num_classes, args=args, logger=logger)
#     if pretrained is not None:
#         state_dict = torch.load(pretrained)
#         model.load_state_dict(state_dict)
#     return model