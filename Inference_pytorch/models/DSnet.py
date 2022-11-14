import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d, QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
# define DepthSeperabelConv2d with depthwise+pointwise
name=0
class DepthSeperabelConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1,padding=1, groups=3, dilation=1, args=None, logger=None):
        super().__init__()
        if args.mode == "WAGE":
            self.depthwise = nn.Sequential( QConv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=padding,groups=in_planes, # padding=dilation -->padding=padding
                         logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='Depthwise' + '_' + str(name) + '_', model=args.model),
                                             nn.BatchNorm2d(in_planes),
                                             nn.ReLU()
                                             )

            self.pointwise = nn.Sequential(QConv2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, logger=logger, wl_input=args.wl_activate,
                         wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target,
                         name='Pointwise' + '_' + str(name) + '_', model=args.model),
                                            nn.BatchNorm2d(out_planes),
                                            nn.ReLU()
                                           )
        elif args.mode == "FP":
            self.depthwise = nn.Sequential( FConv2d(in_planes, in_planes, kernel_size=3, stride=stride,
                             padding=padding, groups=in_planes, bias=False, dilation=dilation,
                             logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight,
                             inference=args.inference,
                             onoffratio=args.onoffratio, cellBit=args.cellBit,
                             subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                             detect=args.detect, target=args.target, cuda=args.cuda,
                             name='Depthwise' + '_' + str(name) + '_'),
                                    nn.BatchNorm2d(in_planes),
                                    nn.ReLU()
                                        )
            self.pointwise = nn.Sequential( FConv2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, bias=False,
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='Pointwise' + '_' + str(name) + '_'),
                                            nn.BatchNorm2d(out_planes),
                                            nn.ReLU()
                                            )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x




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
        linear = FLinear(in_planes, out_planes, bias=True,  # Flase to True
                         logger=logger, wl_input=args.wl_activate, wl_weight=args.wl_weight, inference=args.inference,
                         onoffratio=args.onoffratio, cellBit=args.cellBit,
                         subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                         detect=args.detect, target=args.target, cuda=args.cuda,
                         name='FC' + '_' + str(name) + '_')
    name += 1
    return linear

class MobileNet(nn.Module):
    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """
    def __init__(self, class_num=10, args=None, logger=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv2d(3, 32, 3, stride=1, padding=1, args=args, logger=logger),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.mobilebone = nn.Sequential(
            DepthSeperabelConv2d(32, 64, 1, args=args, logger=logger),
            DepthSeperabelConv2d(64, 128, 2, args=args, logger=logger),
            DepthSeperabelConv2d(128, 128, 1, args=args, logger=logger),
            DepthSeperabelConv2d(128, 256, 2, args=args, logger=logger),
            DepthSeperabelConv2d(256, 256, 1, args=args, logger=logger),
            DepthSeperabelConv2d(256, 512, 2, args=args, logger=logger),
            DepthSeperabelConv2d(512, 512, 1, args=args, logger=logger),
            # DepthSeperabelConv2d(512, 512, 1, args=args, logger=logger),
            # DepthSeperabelConv2d(512, 512, 1, args=args, logger=logger),
            # DepthSeperabelConv2d(512, 512, 1, args=args, logger=logger),
            # DepthSeperabelConv2d(512, 512, 1, args=args, logger=logger),
            # DepthSeperabelConv2d(512, 1024, 2, args=args, logger=logger),
            # DepthSeperabelConv2d(1024, 1024, 1, args=args, logger=logger),
        )
        self.conv2 = nn.Sequential(

        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.avg = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.fc1 = Linear(14400, 1024, args=args, logger=logger)
        self.fc = Linear(64, class_num, args=args, logger=logger)
    def forward(self, x):
        # x = self.conv1(x)
        # print('^^^^^^^^^^^after the  conv1 mbv1 is ', x.shape)
        x = self.mobilebone(x)
        # print('^^^^^^^^^^^after the  conv2 mbv1 is ', x.shape)
        x = self.avg(x)
        # print('____________________________after avgpool2d is ', x.shape)
        x = x.view(x.size(0), -1)
        # print('____________________________before fc is ', x.shape)
        x = self.fc(x)
        # print('____________________________after fc is ', x.shape)

        return x

def MobileNetV1(class_num=10, args=None, logger=None, pretrained=None):
    model = MobileNet(class_num,args = args, logger=logger)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)
    return model