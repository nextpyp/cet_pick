from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
# from cet_pick.DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def fill_up_weights_3d(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    # print('f', f)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            for k in range(w.size(4)):
                w[0, 0, i, j, k] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))*(1 - math.fabs(k / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :, :] = w[0, 0, :, :, :] 

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        # self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        # self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

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


class TomoResClassifier3D(nn.Module):
    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads 
        self.deconv_with_bias = False 

        super(TomoResClassifier3D, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3,3,3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.deconv_layers = self._make_deconv_layer(
        #     4,
        #     [256, 128, 64, 32],
        #     [4, 4, 4, 4],
        # )
        # self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)
        self.feature_3d = nn.Sequential(nn.Conv3d(256*block.expansion, 256*block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256*block.expansion, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
            )
        fill_fc_weights(self.feature_3d)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(256*block.expansion, head_conv)
        fill_fc_weights(self.fc)
        # self.layer_3d = nn.Conv3d(256*block.expansion, 256*block.expansion, kernel_size=3, padding=1, bias=False)
        # self.bn3d = nn.BatchNorm3d(256*block.expansion, momentum=BN_MOMENTUM)
        # self.relu3d = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv1d(1, 1, kernel_size = 3, padding = 1, bias=False)
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


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
                # nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation = dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation = dilation))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1, 
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward_test(self, x1):
        b, c, d, h, w = x1.shape
        
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        bb, dd, ch, hh, ww = x1.shape
        
        x1 = self.feature_3d(x1)
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

    def forward(self,x1, x2):
        #input is batch*1 * d * h * w
        #reshape it to 64 * 1 * 512 * 512 to treat each slice individually


        b, c, d, h, w = x1.shape

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        # x1 = self.layer4(x1)
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        # x2 = self.layer4(x2)
        bb, dd, ch, hh, ww = x1.shape
       
        x1 = self.feature_3d(x1)
        x1 = self.avgpool(x1)

        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.fc(x1)
        x2 = self.feature_3d(x2)
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

            # ret[head] = self.__getattr__(head)(x)


        return [ret1, ret2]


    def _load_pretrained(self, state_dict, inchans=3):
        if inchans == 1:
            conv1_weights = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weights.sum(dim=1, keepdim=True)
        elif inchans != 3:
            assert False, "Invalid number of in channels"

        model_state_dict = self.state_dict()
        msg = 'If you see this, your model does not fully load the ' + \
            'pre-trained weight. The current model only uses partial pre-trained weight and it is ok '
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

    def init_weights(self, num_layers):
        if 1:
            checkpoint = torch.load('/hpc/home/qh36/research/qh36/3D_picking/cet_pick/cet_pick/models/resnet_18.pth')
            pretrained_state_dict = checkpoint['state_dict']

            for k in pretrained_state_dict.copy():
                if k.startswith('module') and not k.startswith('module_list'):
                    pretrained_state_dict[k[7:]] = pretrained_state_dict[k]
                else:
                    pretrained_state_dict[k] = pretrained_state_dict[k]
            self._load_pretrained(pretrained_state_dict, inchans=1)
            print('=> init deconv weights from normal distribution')
            # for name, m in self.deconv_layers.named_modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_simsiam_net_small_3d(num_layers, heads, head_conv = 32):
    block_class, layers = resnet_spec[num_layers]

    model = TomoResClassifier3D(block_class, layers, heads, head_conv=0)
    # model.init_weights(num_layers)
    return model


