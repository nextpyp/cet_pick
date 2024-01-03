'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
import logging

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.001)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class TomoResSmallClassifier(nn.Module):
    def __init__(self, block, layers, heads, head_conv):
        super(TomoResSmallClassifier, self).__init__()
        self.backbone = ResNet(block, layers)
        self.heads = heads
        # input_dim_to_3d = self.backbone
        # self.feature_3d = nn.Sequential(nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm3d(128, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True)
        #     )

        # fill_fc_weights(self.feature_3d)
        # self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, head_conv)
        fill_fc_weights(self.fc)
        for head in self.heads:
            classes = self.heads[head]
            if 'proj' in head:
                fc = nn.Sequential(nn.Linear(head_conv, head_conv, bias=False),
                                   nn.BatchNorm1d(head_conv),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(head_conv, head_conv, bias=False),
                                   nn.BatchNorm1d(head_conv),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(head_conv, head_conv, bias=False),
                                   nn.BatchNorm1d(head_conv, affine=False))
            if 'pred' in head:
                fc = nn.Sequential(nn.Linear(head_conv, head_conv, bias=False),
                                   nn.BatchNorm1d(head_conv),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(head_conv, head_conv)
                    )
            fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _channel_wise_reshape(self, x):
        # resahpe feature map x 
        # input dimension 64 * 1 * 128 * 128
        x = x.permute(1,0,2,3)
        x = x.view(x.shape[0],x.shape[1], -1)
        x = x.permute(2,0,1)
        return x 

    def _reverse_reshape(self, x):
        # reshape it back to correct dimension
        # input dimension is (128*128) * 1 * 64
        x = x.permute(1, 2, 0)
        x = x.view(x.shape[0], x.shape[1], int(x.shape[-1]**(0.5)), int(x.shape[-1]**(0.5)))
        return x 

    def forward_test(self, x1):
        if len(x1.shape) > 4:
            x1 = x1.squeeze(dim=1)
        b, d, h, w = x1.shape
        x1 = self.backbone(x1)
        x1 = self.avgpool(x1)
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.fc(x1)
        ret1 = {}
        for head in self.heads:
            if 'proj' in head:
                z1 = self.__getattr__(head)(x1)
                ret1[head] = z1.detach() 
            if 'pred' in head:
                p1 = self.__getattr__(head)(z1)
                ret1[head] = p1 
        return ret1
        
    def forward(self, x1, x2):
        if len(x1.shape) > 4:
            x1 = x1.squeeze(dim=1)
        if len(x2.shape) > 4:
            x2 = x2.squeeze(dim=1)
        b, d, h, w = x1.shape
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        dd, ch, hh, ww = x1.shape
        x1 = self.avgpool(x1)
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.fc(x1)
        x2 = self.avgpool(x2)
        x2 = x2.reshape(x2.size(0), -1)
        x2 = self.fc(x2)
        ret1 = {}
        ret2 = {}
        for head in self.heads:
            if 'proj' in head:
                z1 = self.__getattr__(head)(x1)
                z2 = self.__getattr__(head)(x2)
                ret1[head] = z1.detach() 
                ret2[head] = z2.detach() 
            if 'pred' in head:
                p1 = self.__getattr__(head)(z1)
                p2 = self.__getattr__(head)(z2)
                ret1[head] = p1 
                ret2[head] = p2

        return [ret1, ret2]

    def _load_pretrained(self, checkpoint, inchans = 1):
        state_dict ={}

        state_dict_ = checkpoint
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict['backbone.'+k[7:]] = state_dict_[k]
            else:
                state_dict['backbone.'+k] = state_dict_[k]
        if inchans == 1:
            conv1_weights = state_dict['backbone.conv1.weight']
            state_dict['backbone.conv1.weight'] = conv1_weights.sum(dim=1, keepdim=True)
        elif inchans != 3:
            assert False, "Invalid number of in channels"
        model_state_dict = self.state_dict()
        msg = 'If you see this, your model does not fully load the ' + \
            'pre-trained weight. Please make sure ' + \
            'you have correctly specified --arch xxx ' + \
            'or set the correct --num_classes for your own dataset.'

        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, '\
                          'loaded shape{}. {}'.format(
                      k, model_state_dict[k].shape, state_dict[k].shape, msg))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k) + msg)
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k) + msg)
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)


resnet_spec = {20: (BasicBlock, [3,3,3]),
               32: (BasicBlock, [5,5,5]),}

def get_simsiam_resnet_small(num_layers, heads, head_conv=256, last_k = 0):
    block_class, layers = resnet_spec[num_layers]
    model = TomoResSmallClassifier(block_class, layers, heads, head_conv)

    return model 
