from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
# from cet_pick.DCNv2.dcn_v2 import DCN
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from cet_pick.models.utils import insize_from_outsize_xyz

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


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__()

        if 'pooling' in kwargs:
            pooling = kwargs['pooling']
            if pooling == 'max':
                kwargs['pooling'] = MaxPool

        if 'heads' in kwargs:
            self.heads = kwargs['heads']

        modules = self.make_modules(**kwargs)
        self.features2d = nn.Sequential(*modules[:-2])
        self.features3d = nn.Sequential(*modules[-2:])
        self.width_xy, self.width_z = insize_from_outsize_xyz(modules, 1, 1)
        self.pad = False 
        # self.feature_head = nn.Sequential(nn.Conv3d(32, head_conv, kernel_size = (3,1,1), padding=(0,0,0), bias=True), 
        #             nn.ReLU(inplace=True)
        #     )

        


        fill_fc_weights(self.features3d)
        fill_fc_weights(self.features2d)
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Conv3d(self.latent_dim, classes, kernel_size=1,stride=1,padding=0, bias=True)
            if 'hm' in head:
                fc.bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)

            self.__setattr__(head, fc)

    
    def fill(self, stride_z=1, stride_xy = 1):
        for mod in self.features2d.children():
            if hasattr(mod, 'fill'):
                stride_xy *= mod.fill(stride_xy)
        # for mod in self.features3d.children():
        #     if hasattr(mod, 'fill'):
        #         stride_z *= mod.fill(stride_z)
        # stride_xy=1
        for mod in self.features3d.children():
            if hasattr(mod, 'fill'):
                # print('mod fill', mod.fill(stride_z, stride_xy)[0])
                (fill_z, fill_xy) = mod.fill(stride_z, stride_xy)
                # print('fill_z', fill_z)
                # print('fill_xy', fill_xy)
                stride_z, stride_xy = stride_z*(fill_z), stride_xy*(fill_xy)
        self.pad = True
        return (stride_z, stride_xy)

    def unfill(self):
        for mod in self.features2d.children():
            if hasattr(mod, 'unfill'):
                mod.unfill()
        for mod in self.features3d.children():
            if hasattr(mod, 'unfill'):
                mod.unfill()
        self.pad = False


    def set_padding(self, pad):
        self.pad = pad

    def forward(self, x):
        if len(x.shape) > 4:
            x = x.squeeze()
        b, d, h, w = x.shape
        # print('batch size', b)
        if b > 1:
            x = x.reshape((-1,h,w)).contiguous()
            x = x.unsqueeze(1)
        else:
            x = x.permute(1,0,2,3)
        # print('after permute size', x.shape)


        if self.pad:
            pxy = self.width_xy//2
            x = F.pad(x, (pxy,pxy,pxy,pxy))
        # print('after pad size', x.shape)
        x = self.features2d(x)
        dd, ch, hh, ww = x.shape 
        if b > 1:
            x = x.reshape((b, d, ch, hh, ww)).contiguous()
            x = x.permute(0,2,1,3,4)
        else:
            x = x.permute(1,0,2,3)
            x = x.unsqueeze(0)
        if self.pad:
            pz = self.width_z//2 
            x = F.pad(x, (0,0,0,0,pz,pz))
            # print('padded', x.shape)
        x = self.features3d(x)
        ret = {}
        for head in self.heads:
            out = self.__getattr__(head)(x)
            if 'proj' in head:
                out = torch.nn.functional.normalize(out, dim=1)
            ret[head] = out
        return [ret]



class ResidA(nn.Module):
    dim = 2 
    def __init__(self, nin, nhidden, nout, dilation=1, stride=1
                , activation=nn.ReLU, bn=False):
        

        super(ResidA, self).__init__()

        self.bn = bn
        bias = not bn

        if nin != nout:
            self.proj = nn.Conv2d(nin, nout, 1, stride=stride, bias=False)
        
        self.conv0 = nn.Conv2d(nin, nhidden, 3, bias=bias)
        if self.bn:
            self.bn0 = nn.BatchNorm2d(nhidden)
        self.act0 = activation(inplace=True)

        self.conv1 = nn.Conv2d(nhidden, nout, 3, dilation=dilation, stride=stride
                              , bias=bias)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(nout)
        self.act1 = activation(inplace=True)

        self.kernel_size = 2*dilation + 3
        self.stride = stride
        self.dilation = 1
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            self.conv0.padding = (1,1)
            self.conv1.padding = self.conv1.dilation
            self.padding = self.kernel_size//2
        else:
            self.conv0.padding = (0,0)
            self.conv1.padding = (0,0)
            self.padding = 0

    def fill(self, stride):
        self.conv0.dilation = (stride, stride)
        self.conv0.padding = (self.conv0.padding[0]*stride, self.conv0.padding[1]*stride)
        self.conv1.dilation = (self.conv1.dilation[0]*stride, self.conv1.dilation[1]*stride)
        self.conv1.stride = (1,1)
        self.conv1.padding = (self.conv1.padding[0]*stride, self.conv1.padding[1]*stride)
        if hasattr(self, 'proj'):
            self.proj.stride = (1,1)
        self.dilation = self.dilation*stride
        return self.stride

    def unfill(self):
        self.conv0.dilation = (1,1)
        self.conv0.padding = (self.conv0.padding[0]//self.dilation, self.conv0.padding[1]//self.dilation)
        self.conv1.dilation = (self.conv1.dilation[0]//self.dilation, self.conv1.dilation[1]//self.dilation)
        self.conv1.stride = (self.stride,self.stride)
        self.conv1.padding = (self.conv1.padding[0]//self.dilation, self.conv1.padding[1]//self.dilation)
        if hasattr(self, 'proj'):
            self.proj.stride = (self.stride,self.stride)
        self.dilation = 1

    def forward(self, x):

        h = self.conv0(x)
        if self.bn:
            h = self.bn0(h)
        h = self.act0(h)

        y = self.conv1(h)

        #d2 = x.size(2) - y.size(2)
        #d3 = x.size(3) - y.size(3)
        #if d2 > 0 or d3 > 0:
        #    lb2 = d2//2
        #    ub2 = d2 - lb2
        #    lb3 = d3//2
        #    ub3 = d3 - lb3
        #    x = x[:,:,lb2:-ub2,lb3:-ub3]

        edge = self.conv0.dilation[0] + self.conv1.dilation[0]
        x = x[:,:,edge:-edge,edge:-edge]

        if hasattr(self, 'proj'):
            x = self.proj(x)
        elif self.conv1.stride[0] > 1:
            x = x[:,:,::self.stride,::self.stride]
        

        y = y + x
        if self.bn:
            y = self.bn1(y)
        y = self.act1(y)

        return y


class ResNet8(ResNet):
    def make_modules(self, units=[32, 64, 128], bn=True, dropout=0.0
                    , activation=nn.ReLU, pooling=None, **kwargs):
        if units is None:
            units = [32, 64, 128]
        elif type(units) is not list:
            units = int(units)
            units = [units, 2*units, 4*units]

        self.num_features = units[-1]
        self.stride = 1
        if pooling is None:
            self.stride = 2
        stride = self.stride
        if 'head_conv' in kwargs:
            self.latent_dim = kwargs['head_conv']
        modules = [
                BasicConv2d(1, units[0], 7, stride=stride, bn=bn, activation=activation),
                ]
        if pooling is not None:
            modules.append(pooling(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[0], units[0], units[0], dilation=2, bn=bn, activation=activation),
                ResidA(units[0], units[0], units[1], dilation=2
                      , stride=stride, bn=bn, activation=activation),
                ]
        if pooling is not None:
            modules.append(pooling(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[1], units[1], units[1], dilation=2, bn=bn, activation=activation),
                BasicConv3d(units[1], units[2], (5,5,5), bn=bn, dilation = (1,1,1), activation=activation),
                BasicConv3d(units[2], self.latent_dim, (1,1,1), bn=bn, dilation = (1,1,1), activation=activation)
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        # self.latent_dim = units[-1]

        return modules

class BasicConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, dilation=(1,1,1), stride=(1,1,1), bn=False, activation=nn.ReLU):
        super(BasicConv3d, self).__init__()
        bias = not bn 
        self.conv = nn.Conv3d(nin, nout, kernel_size, dilation=dilation, stride=stride, bias=bias)
        if bn:
            self.bn = nn.BatchNorm3d(nout)

        self.act = activation(inplace=True)
        self.kernel_size_z, self.kernel_size_xy = kernel_size[0], kernel_size[1]
        self.kernel_size = kernel_size
        self.stride = stride
        self.stride_z, self.stride_xy = stride[0], stride[1]
        self.dilation = dilation 
        self.dilation_z, self.dilation_xy = dilation[0], dilation[1]
        self.og_dilation_z, self.og_dilation_xy = self.dilation_z, self.dilation_xy
        self.padding_z, self.padding_xy = 0, 0
        self.padding = (self.padding_z, self.padding_xy, self.padding_xy)

    def set_padding(self, pad):
        if pad:
            p_z = self.dilation_z*(self.kernel_size_z//2)
            p_xy = self.dilation_xy*(self.kernel_size_xy//2)
            self.conv.padding = (p_z, p_xy, p_xy)
            self.padding_z, self.padding_xy = p_z, p_xy
        else:
            self.conv.padding = (0,0,0)
            self.padding_z, self.padding_xy = 0, 0
    def fill(self, stride_z, stride_xy):
        # print('fill stride_z', stride_z)
        # print()
        self.conv.dilation = (self.og_dilation_z * stride_z, self.og_dilation_xy*stride_xy, self.og_dilation_xy*stride_xy)
        self.conv.stride = (1,1,1)
        self.conv.padding = (self.conv.padding[0]*stride_z, self.conv.padding[1]*stride_xy, self.conv.padding[2]*stride_xy)
        self.dilation_z, self.dilation_xy = self.dilation_z * stride_z, self.dilation_xy * stride_xy
        return (self.stride_z, self.stride_xy)

    def unfill(self):
        # print('self.og_dilation', self.og_dilation_z)
        stride_z, stride_xy = self.dilation_z//self.og_dilation_z, self.dilation_xy//self.og_dilation_xy
        self.conv.dilation = (self.og_dilation_z, self.og_dilation_xy, self.og_dilation_xy)
        self.conv.stride = (self.stride_z,self.stride_xy, self.stride_xy)
        self.conv.padding = (self.conv.padding[0]//stride_z, self.conv.padding[1]//stride_xy, self.conv.padding[2]//stride_xy)
        self.dilation_z, self.dilation_xy = self.og_dilation_z, self.og_dilation_xy

    def forward(self, x):
        # print('conv in', x.shape)
        y = self.conv(x)
        if hasattr(self, 'bn'):
            y = self.bn(y)
        # print('conv out', y.shape)
        return self.act(y)

# class BasicConv3d(nn.Module):
#     dim = 3
#     def __init__(self, nin, nout, kernel_size, dilation=1, stride=(1,1,1)
#                 , bn=False, activation=nn.ReLU):
#         super(BasicConv3d, self).__init__()

#         bias = not bn
#         self.conv = nn.Conv3d(nin, nout, kernel_size, dilation=dilation
#                              , stride=stride, bias=bias)
#         if bn:
#             self.bn = nn.BatchNorm3d(nout)
#         self.act = activation(inplace=True)

#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.dilation = dilation
#         print('dilation.....', dilation)
#         self.og_dilation = dilation
#         self.padding = (0,0,0)

#     def set_padding(self, pad):
#         if pad:
#             p = self.dilation*(self.kernel_size[1]//2)
#             pz = self.dilation*(self.kernel_size[0]//2)
#             self.conv.padding = (pz,p, p)
#             self.padding = (pz,p,p)
#         else:
#             self.conv.padding = (0,0,0)
#             self.padding = (0,0,0)

#     def fill(self, stride):
#         print('input stride', stride)
#         print('input dilation', self.og_dilation)
#         # self.conv.dilation = (self.og_dilation[0]*stride, self.og_dilation[1]*stride, self.og_dilation[2]*stride)
#         self.conv.dilation = (self.og_dilation*stride,self.og_dilation*stride,self.og_dilation*stride)
#         self.conv.stride = (1,1,1)
#         self.conv.padding = (self.conv.padding[0]*stride,self.conv.padding[1]*stride, self.conv.padding[2]*stride)
#         # self.dilation *= stride
#         # self.dilation = (self.dilation[0]*stride, self.dilation[1]*stride, self.dilation[2]*stride)
#         self.dilation *= stride
#         return self.stride

#     def unfill(self):
#         # stride = (self.dilation[0]//self.og_dilation[0],self.dilation[1]//self.og_dilation[1],self.dilation[1]//self.og_dilation[1])
#         stride = self.dilation//self.og_dilation
#         self.conv.dilation = (self.og_dilation, self.og_dilation, self.og_dilation)
#         self.conv.stride = (self.stride, self.stride, self.stride)
#         self.conv.padding = (self.conv.padding[0]//stride,self.conv.padding[1]//stride, self.conv.padding[2]//stride)
#         self.dilation = self.og_dilation

#     def forward(self, x):
#         print('input x', x.shape)
#         y = self.conv(x)
#         print('output x', x.shape)
#         if hasattr(self, 'bn'):
#             y = self.bn(y)
#         return self.act(y)


class BasicConv2d(nn.Module):
    dim = 2
    def __init__(self, nin, nout, kernel_size, dilation=1, stride=1
                , bn=False, activation=nn.ReLU):
        super(BasicConv2d, self).__init__()

        bias = not bn
        self.conv = nn.Conv2d(nin, nout, kernel_size, dilation=dilation
                             , stride=stride, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(nout)
        self.act = activation(inplace=True)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.og_dilation = dilation
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            p = self.dilation*(self.kernel_size//2)
            self.conv.padding = (p, p)
            self.padding = p
        else:
            self.conv.padding = (0,0)
            self.padding = 0

    def fill(self, stride):
        self.conv.dilation = (self.og_dilation*stride, self.og_dilation*stride)
        self.conv.stride = (1,1)
        self.conv.padding = (self.conv.padding[0]*stride, self.conv.padding[1]*stride)
        self.dilation *= stride
        return self.stride

    def unfill(self):
        stride = self.dilation//self.og_dilation
        self.conv.dilation = (self.og_dilation, self.og_dilation)
        self.conv.stride = (self.stride,self.stride)
        self.conv.padding = (self.conv.padding[0]//stride, self.conv.padding[1]//stride)
        self.dilation = self.og_dilation

    def forward(self, x):
        y = self.conv(x)
        if hasattr(self, 'bn'):
            y = self.bn(y)
        return self.act(y)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
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
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.001)


# class TomoResClassifier(nn.Module):
#     def __init__(self, block, layers, heads, head_conv):
#         self.inplanes = 64
#         self.heads = heads 
#         self.deconv_with_bias = False 

#         super(TomoResClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         # self.deconv_layers = self._make_deconv_layer(
#         #     4,
#         #     [256, 128, 64, 32],
#         #     [4, 4, 4, 4],
#         # )
#         # self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)
#         # self.conv3 = nn.Conv1d(1, 1, kernel_size = 3, padding = 1, bias=False)
#         for head in self.heads:
#             classes = self.heads[head]
#             if head_conv > 0:
#                 fc = nn.Sequential(
#                     nn.Conv3d(32, head_conv, kernel_size = 3, padding=1, bias=True), 
#                     nn.ReLU(inplace=True),
#                     nn.Conv3d(head_conv, classes, kernel_size = 1, stride=1, padding=0, bias=True))
#                 if 'hm' in head:
#                     fc[-1].bias.data.fill_(-2.19)
#                 else:
#                     fill_fc_weights(fc)
#             else:
#                 # fc = nn.Conv3d(1, classes, kernel_size = 1, stride = 1, padding=0, bias=True)
#                 fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
#                     nn.Flatten(),
#                     nn.Linear(128, 64),
#                     nn.ReLU(),
#                     nn.Dropout(0.5),
#                     nn.Linear(64, 1)
#                     )
#                 if 'hm' in head:
#                     fc[-1].bias.data.fill_(-2.19)
#                 else:
#                     fill_fc_weights(fc)
#             self.__setattr__(head, fc)




#     def _channel_wise_reshape(self, x):
#         # resahpe feature map x 
#         # input dimension 64 * 1 * 128 * 128
#         x = x.permute(1,0,2,3)
#         x = x.view(x.shape[0],x.shape[1], -1)
#         x = x.permute(2,0,1)
#         return x 

#     def _reverse_reshape(self, x):
#         # reshape it back to correct dimension
#         # input dimension is (128*128) * 1 * 64
#         x = x.permute(1, 2, 0)
#         x = x.view(x.shape[0], x.shape[1], int(x.shape[-1]**(0.5)), int(x.shape[-1]**(0.5)))
#         return x 


#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 # nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0

#         return deconv_kernel, padding, output_padding

#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'

#         layers = []
#         for i in range(num_layers):
#             kernel, padding, output_padding = \
#                 self._get_deconv_cfg(num_kernels[i], i)

#             planes = num_filters[i]
#             fc = DCN(self.inplanes, planes, 
#                     kernel_size=(3,3), stride=1,
#                     padding=1, dilation=1, deformable_groups=1)
#             # fc = nn.Conv2d(self.inplanes, planes,
#             #         kernel_size=3, stride=1, 
#             #         padding=1, dilation=1, bias=False)
#             # fill_fc_weights(fc)
#             up = nn.ConvTranspose2d(
#                     in_channels=planes,
#                     out_channels=planes,
#                     kernel_size=kernel,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=self.deconv_with_bias)
#             fill_up_weights(up)

#             layers.append(fc)
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(up)
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             self.inplanes = planes

#         return nn.Sequential(*layers)

#     def forward(self,x):
#         #input is 1 * 64 * 512 * 512 
#         #reshape it to 64 * 1 * 512 * 512 to treat each slice individually
#         x = x.permute(1,0,2,3)
#         # print('x', x.shape)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         # x = self.layer3(x)
#         # x = self.layer4(x)
#         # print('x', x.shape)
#         y = nn.AdaptiveAvgPool2d(1)(x)
#         # print('y', y.shape)
#         # x = self.deconv_layers(x)
#         # now the output shape should be 64 * 64 * 128 * 128
#         # we do a covolution so it becomes 64 * 1 * 128 * 128
#         # print('deconv_x', x.shape)
#         # x = x.permute(1,0,2,3)
#         # x = x.unsqueeze(0)
#         # x = self.conv2(x)
#         # # x = x.permute(1,0,2,3)
#         # # print('conv2', x.shape)
#         # x = self._channel_wise_reshape(x)
#         # # # print('reshape_x', x.shape)
#         # x = self.conv3(x)
#         # # # print('single conv,', x.shape)
#         # x = self._reverse_reshape(x)
#         # # print('reverse', x.shape)
#         # x = x.unsqueeze(1)
#         ret = {}
#         for head in self.heads:
#             ret[head] = self.__getattr__(head)(x)
#             # print('head', head)
#             # print('out', ret[head].shape)

#         return [ret]


#     def _load_pretrained(self, state_dict, inchans=3):
#         if inchans == 1:
#             conv1_weights = state_dict['conv1.weight']
#             state_dict['conv1.weight'] = conv1_weights.sum(dim=1, keepdim=True)
#         elif inchans != 3:
#             assert False, "Invalid number of in channels"
#         self.load_state_dict(state_dict, strict=False)
#     def init_weights(self, num_layers):
#         if 1:
#             url = model_urls['resnet{}'.format(num_layers)]
#             pretrained_state_dict = model_zoo.load_url(url)
#             print('=> loading pretrained model {}'.format(url))
#             # self.load_state_dict(pretrained_state_dict, strict=False)
#             self._load_pretrained(pretrained_state_dict, inchans=1)
#             print('=> init deconv weights from normal distribution')
#             # for name, m in self.deconv_layers.named_modules():
#             #     if isinstance(m, nn.BatchNorm2d):
#             #         nn.init.constant_(m.weight, 1)
#             #         nn.init.constant_(m.bias, 0)

# resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
#                34: (BasicBlock, [3, 4, 6, 3]),
#                50: (Bottleneck, [3, 4, 6, 3]),
#                101: (Bottleneck, [3, 4, 23, 3]),
#                152: (Bottleneck, [3, 8, 36, 3])}

def get_tomo_class_net_small(num_layers, heads, head_conv = 32, last_k=5):
    # block_class, layers = resnet_spec[num_layers]

    # model = TomoResClassifier(block_class, layers, heads, head_conv=0)
    print('using new classifier model....')
    latent_dim = heads['proj']
    # print('latent dim', latent_dim)
    model = ResNet8(bn=False, heads=heads, head_conv = latent_dim)
    # model.init_weights(num_layers)
    return model


