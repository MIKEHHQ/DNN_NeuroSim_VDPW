'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.quantization_cpu_np_infer import QConv2d, QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
from torchvision.models.utils import load_state_dict_from_url
name=0

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,padding=1, args=None, logger=None):
    """3x3 convolution with padding"""
    global name
    if args.mode == "WAGE":
        conv2d = QConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, # padding=dilation -->padding=padding
                         logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='Conv3x3' + '_' + str(name) + '_', model=args.model)

    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=padding, groups=groups, bias=False, dilation=dilation,
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='Conv3x3' + '_' + str(name) + '_')
    name += 1
    return conv2d


def conv1x1(in_planes, out_planes, stride=1,padding=0, args=None, logger=None):
    """1x1 convolution"""
    global name
    if args.mode == "WAGE":
        conv2d = QConv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=padding, logger=logger, wl_input=args.wl_activate,
                         wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='Conv1x1' + '_' + str(name) + '_', model=args.model)
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=padding, bias=False,
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='Conv1x1' + '_' + str(name) + '_')
    name += 1
    return conv2d


def Conv2d(in_planes, out_planes, kernel_size, stride, padding, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":
        conv2d = QConv2d(in_planes, out_planes, kernel_size, stride, padding, logger=logger, wl_input=args.wl_activate,
                         wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='Conv' + '_' + str(name) + '_', model=args.model)
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False,
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
        linear = FLinear(in_planes, out_planes, bias=False,
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='FC' + '_' + str(name) + '_')
    name += 1
    return linear






class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride,args=None, logger=None):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = conv1x1(in_planes, planes,stride=1,padding=0, args=args, logger=logger)
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, padding=1, groups=planes, args=args, logger=logger)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, out_planes, stride=1, padding=0, args=args, logger=logger)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, args=args, logger=logger),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, args=None, logger=None, **kwargs):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1, args=args, logger=logger)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, args=args, logger=logger)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes, args, logger):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, args=args, logger=logger))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
