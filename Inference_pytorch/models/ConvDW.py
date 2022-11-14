import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d, QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
name=0
def Conv2d(in_channel, out_planes, kernel_size, stride, padding, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":
        conv2d = QConv2d(in_channel, out_planes, kernel_size, stride, padding, logger=logger, wl_input=args.wl_activate,
                         wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='Conv' + '_' + str(name) + '_', model=args.model)
    elif args.mode == "FP":
        conv2d = FConv2d(in_channel, out_planes, kernel_size, stride, padding, bias=False,
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='Conv' + '_' + str(name) + '_')
    name += 1
    return conv2d


def Linear(in_channel, out_planes, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":
        linear = QLinear(in_channel, out_planes,
                         logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error,
                         wl_weight=args.wl_weight, inference=args.inference, onoffratio=args.onoffratio,
                         cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='FC' + '_' + str(name) + '_', model=args.model)
    elif args.mode == "FP":
        linear = FLinear(in_channel, out_planes, bias=True,  # Flase to True
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='FC' + '_' + str(name) + '_')
    name += 1
    return linear


class MobleNetV1(nn.Module):
    def __init__(self, num_classes):

        super(MobleNetV1, self).__init__()
        self.mobilebone = nn.Sequential(
            self._conv_bn(3, 32, 2),
            self._conv_dw(32, 64, 1),
            self._conv_dw(64, 128, 2),
            self._conv_dw(128, 128, 1),
            self._conv_dw(128, 256, 2),
            self._conv_dw(256, 256, 1),
            self._conv_dw(256, 512, 2),
            self._top_conv(512, 512, 5),
            self._conv_dw(512, 1024, 2),
            self._conv_dw(1024, 1024, 1),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(1024, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.mobilebone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        prob = F.softmax(x)
        return x, prob

    def _top_conv(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def _conv_bn(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def _conv_dw(self, in_channel, out_channel, stride,padding,dilation,args=None, logger=None):
        return nn.Sequential(
            FConv2d(in_channel, in_channel, kernel_size=3, stride=stride,
                    padding=padding, groups=in_channel, bias=False, dilation=dilation,
                    logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight,
                    inference=args.inference,
                    onoffratio=args.onoffratio, cellBit=args.cellBit,
                    subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                    detect=args.detect, target=args.target, cuda=args.cuda,
                    name='Depthwise' + '_' + str(name) + '_'),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
        )