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

def conv_S(in_planes,out_planes,stride=1,padding=1):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(in_planes,out_planes,kernel_size=(1,3,3),stride=1,
                     padding=padding,bias=False)

def conv_T(in_planes,out_planes,stride=1,padding=1):
    # conv T is 3x1x1
    return nn.Conv3d(in_planes,out_planes,kernel_size=(3,1,1),stride=1,
                     padding=padding,bias=False)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_3d = False):
        super(Bottleneck, self).__init__()
        stride_p = stride 
        self.downsample = downsample
        if not self.downsample == None:
            stride_p=(1,2,2)

        if is_3d:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p)
            self.conv2 = conv_S(planes, planes, stride=1, padding=(0,1,1))
            self.conv3 = conv_T(planes, planes, stride=1, padding=(1,0,0))
            self.conv4 = nn.Conv3d(planes, planes, kernel_size=1, bias=False)
        else:
            stride_p = 1
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride = stride_p)
            # self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride_p,
                               padding=1, bias=False)
            # self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

            self.conv4 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
            # self.bn4 = nn.BatchNorm2d(planes * self.expansion,
                                  # momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.is_3d = is_3d

    def ST_A(self, x):
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        return x 

    def forward(self, x):
        residual = x 
        # print('res_x', residual.shape)
        
        if self.is_3d:
            out = self.conv1(x)
            out = self.relu(out)
            # print('outcoonv1', out.shape)
            out = self.ST_A(out)
            # print('outsta', out.shape)
            out = self.conv4(out)
            # print('final_out', out.shape)

        else:
            out = self.conv1(x)
            # out = self.bn1(out)
            out = self.relu(out)
            # print('outcoonv1', out.shape)
            out = self.conv2(out)
            # out = self.bn2(out)
            out = self.relu(out)
            # print('outcoonv2', out.shape)
            out = self.conv4(out)
            # print('outcoonv4', out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out 



class Bottleneck3D(nn.Module):
    expansion = 4
    


    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.downsample = downsample 

        stride_p = stride 
        if not self.downsample == None:
            stride_p=(1,2,2)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = conv_S(planes, planes, stride=1, padding=(0,1,1))
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = conv_T(planes, planes, stride=1, padding=(1,0,0))
        self.bn3 = nn.BatchNorm3d(planes)

        self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm3d(planes * 4)


        self.relu = nn.ReLU(inplace=True)

        self.stride = stride 

    def ST_A(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x 

    def forward(x):
        residual = x 

        # 1 * 1 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # doing the 3D info merge operation 

        out = self.ST_A(out)

        # 1 * 1 

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck2D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class TomoP3DNet(nn.Module):
    def __init__(self, block, layers, heads, head_conv, bn = False):
        self.inplanes = 64
        self.heads = heads 
        self.deconv_with_bias = False 

        super(TomoP3DNet, self).__init__()
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
        # # self.bn1 = nn.BatchNorm3d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool3d(kernel_size=(2,3,3), stride=2, padding=(0,1,1))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], is_3d=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, is_3d=True)
        self.bn = bn

        self.deconv_layers = self._make_deconv_layer(
            2,
            [64, 32],
            [4, 4],
        )

        # self.feature_head = nn.Conv2d(32, 16 kernel_size=3, padding=1, bias=True)
        self.feature_head = nn.Sequential(nn.Conv3d(32, 16, kernel_size = (3,3,3), padding=(1,1,1), bias=True), 
                    nn.ReLU(inplace=True)
            )
        fill_fc_weights(self.feature_head)
        fill_fc_weights(self.layer2)

        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Conv3d(16, classes, kernel_size = 1, stride = 1, padding=0, bias=True)
            if 'hm' in head:
                fc.bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
    def _make_layer(self, block, planes, blocks, stride=1, is_3d = False):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            if not is_3d:
                # if self.bn:
                # downsample = nn.Sequential(
                #     nn.Conv2d(self.inplanes, planes* block.expansion, kernel_size=1, stride = 2, bias=False),
                #     nn.BatchNorm2d(planes * block.expansion)
                #     )
                # else:
                downsample = nn.Conv2d(self.inplanes, planes* block.expansion, kernel_size=1, stride = 2, bias=False)
            else:
                # if self.bn:
                #     downsample = nn.Sequential(
                #         nn.Conv3d(self.inplanes, planes*block.expansion, kernel_size=1, stride = (1,2,2), bias=False),
                #         nn.BatchNorm3d(planes * block.expansion)
                #         )
                # else:
                downsample = nn.Conv3d(self.inplanes, planes*block.expansion, kernel_size=1, stride = (1,2,2), bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, is_3d))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_3d = is_3d))

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
            # fc = DCN(self.inplanes, planes, 
            #         kernel_size=(3,3), stride=1,
            #         padding=1, dilation=1, deformable_groups=1)
            fc = nn.Conv2d(self.inplanes, planes,
                    kernel_size=3, stride=1, 
                    padding=1, dilation=1, bias=False)
            fill_fc_weights(fc)
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
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):

        # input is 1 * z * w * h 

        x = x.permute(1,0,2,3)
        # print('x', x.shape)
        x = self.conv1(x)
        # print('xconv1',x.shape) 
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        #z * 64 * w//4 * h //4
        x = x.permute(1,0,2,3)
        x = x.unsqueeze(0)
        x = self.layer2(x)

        #1 * c * z * w//8 * h//8
        x = x.squeeze(0)
        x = x.permute(1,0,2,3)
        # print('reshapex', x.shape)
        x = self.deconv_layers(x)
        x = x.permute(1,0,2,3)
        x = x.unsqueeze(0)
        
        x = self.feature_head(x)
        
        ret = {}

        for head in self.heads:
            # head_out = head + '_out'
            # classes = self.heads[head]
            # print('head', head)
            out = self.__getattr__(head)(x)
            if 'proj' in head:
                # print('out', out.shape)
                # out = out.view(1, 1, 16, -1)
                out = torch.nn.functional.normalize(out, dim=1)
            ret[head] = out
        return [ret]

    def _load_pretrained(self, state_dict, inchans=3):
        if inchans == 1:
            conv1_weights = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weights.sum(dim=1, keepdim=True)
        elif inchans != 3:
            assert False, "Invalid number of in channels"
        self.load_state_dict(state_dict, strict=False)
    def init_weights(self, num_layers):
        if 1:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            # url = model_urls['resnet{}'.format(num_layers)]
            # pretrained_state_dict = model_zoo.load_url(url)
            # print('=> loading pretrained model {}'.format(url))
            # # self.load_state_dict(pretrained_state_dict, strict=False)
            # self._load_pretrained(pretrained_state_dict, inchans=1)
            # print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

resnet_spec = {18: (Bottleneck, [2, 2, 2, 2]),
               34: (Bottleneck, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_tomo_p3d_net_small(num_layers, heads, head_conv = 16):
    print('is using p3d')
    block_class, layers = resnet_spec[num_layers]

    model = TomoP3DNet(block_class, layers, heads, head_conv=16)
    # model.init_weights(num_layers)
    return model







