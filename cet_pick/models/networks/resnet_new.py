from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
from cet_pick.models.utils import insize_from_outsize_3d, out_from_in
import torch.nn.functional as F

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

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
    def __init__(self, heads, head_conv, *args, **kwargs):
        super(ResNet, self).__init__()
        if 'pooling' in kwargs:
            pooling = kwargs['pooling']
            if pooling == 'max':
                kwargs['pooling'] = MaxPool
        modules_all, modules_2d = self.make_modules(**kwargs)
        self.heads = heads
        print('modules:', modules_all)
        self.features_2d = nn.Sequential(*modules_2d)
        self.features_3d = nn.Sequential(*modules_all[-2:])
        print('features_3d', self.features_3d)
        fill_fc_weights(self.features_2d)
        fill_fc_weights(self.features_3d)
        self.width_z, self.width_xy = insize_from_outsize_3d(modules_all, 1, 1)
        self.pad = False 
        for head in self.heads:
            classes = self.heads[head] 
            fc = nn.Conv3d(self.latent_dim, classes, kernel_size = 1, stride = 1, padding=0, bias=True)
            if 'hm' in head:
                fc.bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def fill(self, stride_z=1, stride_xy=1):
        for mod in self.features_2d.children():
            if hasattr(mod, 'fill'):
                stride_xy *= mod.fill(stride_xy)
        stride_xy=1
        for mod in self.features_3d.children():
            if hasattr(mod, 'fill'):
                # print('mod fill', mod.fill(stride_z, stride_xy)[0])
                (fill_z, fill_xy) = mod.fill(stride_z, stride_xy)
                print('fill_z', fill_z)
                print('fill_xy', fill_xy)
                stride_z, stride_xy = stride_z*(fill_z), stride_xy*(fill_xy)
        self.pad = True
        return (stride_z, stride_xy)

    def unfill(self):
        for mod in self.features_2d.children():
            if hasattr(mod, 'unfill'):
                mod.unfill()
        for mod in self.features_3d.children():
            if hasattr(mod, 'unfill'):
                mod.unfill()
        self.pad = False

    def set_padding(self, pad):
        self.pad = pad

    def forward(self, x):
        if len(x.shape) > 4:
            x = x.squeeze()
        if self.pad:
            p_z, p_xy = self.width_z//2, self.width_xy//2
            x_2d = F.pad(x, (p_xy,p_xy,p_xy,p_xy))
        else:
            x_2d = x
        b, d, h, w = x_2d.shape 
        if b > 1:
            x_2d = x_2d.reshape((-1,h,w)).contiguous()
            x_2d = x_2d.unsqueeze(1)
        else:
            x_2d = x_2d.permute(1,0,2,3)

        x_out2d = self.features_2d(x_2d)
        dd, ch, hh, ww = x_out2d.shape 

        if b>1:
            x_out2d = x_out2d.reshape((b, d, ch, hh, ww)).contiguous()
            x_out2d = x_out2d.permute(0,2,1,3,4)
        else:
            x_out2d = x_out2d.permute(1,0,2,3)
            x_out2d = x_out2d.unsqueeze(0)
        x_out3d = self.features_3d(x_out2d)

        ret = {}
        for head in self.heads:
            out = self.__getattr__(head)(x_out3d)
            if 'proj' or 'pretext' in head:
                out = torch.nn.functional.normalize(out, dim=1)
            ret[head] = out 

        return [ret]

class ResNet8_more3D(ResNet):
    def make_modules(self, units=[16, 32, 64], bn=True, dropout=0.0,
                    activation=nn.ReLU, pooling=None, **kwargs):
        if units is None:
            units = [32, 64, 128]
        elif type(units) is not list:
            units = int(units)
            units = [units, 2*units, 4*units]
        self.num_features = units[-1]
        self.stride_z, self.stride_xy = 1, 1
        if pooling is None:
            self.stride_xy = 2
            self.stride_z = 2
        stride_z, stride_xy = self.stride_z, self.stride_xy
        modules = [
                BasicConv2d(1, units[0], 3, stride=stride_xy, bn=bn, activation=activation),
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))

        modules += [
                ResidA(units[0], units[0], units[0], dilation=2, bn=bn, activation=activation),
                ResidA(units[0], units[0], units[1], dilation=2, stride = stride_xy, bn=bn, activation=activation),
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))
        modules += [
                ResidA(units[1], units[1], units[1], dilation=2, bn=bn, activation=activation)]

        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                BasicConv3d(units[1], units[2], (3,3,3), stride= (stride_z,stride_xy,stride_xy), bn=bn, activation=activation),
                BasicConv3d(units[2], units[1], (3,3,3), stride = (1,1,1),bn=bn, activation=activation),
        ]

        self.latent_dim = units[1]
        return modules, modules[:-2]


class ResNet8_last3D(ResNet):
    def make_modules(self, units=[16, 32, 64], bn=True, dropout=0.0,
                    activation=nn.ReLU, pooling=None, **kwargs):
        if units is None:
            units = [32, 64, 128]
        elif type(units) is not list:
            units = int(units)
            units = [units, 2*units, 4*units]
        self.num_features = units[-1]
        self.stride_z, self.stride_xy = 1, 1
        if pooling is None:
            self.stride_xy = 2
            self.stride_z = 2
        stride_z, stride_xy = self.stride_z, self.stride_xy
        modules = [
                BasicConv2d(1, units[0], 7, stride=stride_xy, bn=bn, activation=activation),
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))

        modules += [
                ResidA(units[0], units[0], units[0], dilation=2, bn=bn, activation=activation),
                ResidA(units[0], units[0], units[1], dilation=2, stride = stride_xy, bn=bn, activation=activation),
                ]

        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))
        modules += [
                ResidA(units[1], units[1], units[1], dilation=2, bn=bn, activation=activation),
                BasicConv2d(units[1], units[2], 5, bn=bn, activation=activation)]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [BasicConv3d(units[2], units[2], (3,1,1), stride = (stride_z,1,1),bn=bn, activation=activation),
                    BasicConv3d(units[2], units[1], (3,1,1), stride = (1,1,1), bn=bn, activation=activation)]

        self.latent_dim = units[1]
        return modules, modules[:-2]

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

        modules = [
                BasicConv2d(1, units[0], 5, stride=stride, bn=bn, activation=activation),
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))

        modules += [
                ResidA(units[0], units[0], units[0], dilation=1, bn=bn, activation=activation),
                ResidA(units[0], units[0], units[1], dilation=2, bn=bn, activation=activation),
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[1], units[1], units[1], dilation=1, bn=bn, activation=activation),
                BasicConv2d(units[1], units[2], 3, bn=bn, activation=activation)
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        self.latent_dim = units[-1]

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
        print('fill stride_z', stride_z)
        print()
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
        print('conv in', x.shape)
        y = self.conv(x)
        if hasattr(self, 'bn'):
            y = self.bn(y)
        print('conv out', y.shape)
        return self.act(y)

class BasicConv2d(nn.Module):
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
        # print('conv padding', self.conv.padding)
        return self.stride

    def unfill(self):
        stride = self.dilation//self.og_dilation
        self.conv.dilation = (self.og_dilation, self.og_dilation)
        self.conv.stride = (self.stride,self.stride)
        self.conv.padding = (self.conv.padding[0]//stride, self.conv.padding[1]//stride)
        self.dilation = self.og_dilation

    def forward(self, x):
        print('conv in', x.shape)
        y = self.conv(x)
        if hasattr(self, 'bn'):
            y = self.bn(y)
        print('conv out', y.shape)
        return self.act(y)

class ResidA(nn.Module):
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

        self.kernel_size = 2*dilation+3
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
        self.conv0.dilation = (self.conv0.dilation[0]*stride, self.conv0.dilation[0]*stride)
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
        # print('input', x.shape)
        h = self.conv0(x)
        # print('conv0', h.shape)
        if self.bn:
            h = self.bn0(h)
        h = self.act0(h)

        y = self.conv1(h)
        # print('conv1', y.shape)
        #d2 = x.size(2) - y.size(2)
        #d3 = x.size(3) - y.size(3)
        #if d2 > 0 or d3 > 0:
        #    lb2 = d2//2
        #    ub2 = d2 - lb2
        #    lb3 = d3//2
        #    ub3 = d3 - lb3
        #    x = x[:,:,lb2:-ub2,lb3:-ub3]
        # print('conv0 dilation', conv0.dilation[0])
        edge = self.conv0.dilation[0] + self.conv1.dilation[0]
        # print('edge', edge)
        x = x[:,:,edge:-edge,edge:-edge]
        # print('remove edge x', x.shape)
        if hasattr(self, 'proj'):
            x = self.proj(x)
            print('has project:', x.shape)
        elif self.conv1.stride[0] > 1:
            x = x[:,:,::self.stride,::self.stride]
            print('stride effect', x.shape)
        

        y = y + x
        if self.bn:
            y = self.bn1(y)
        y = self.act1(y)
        return y 

def get_resnet_new(num_layers, heads, head_conv = 0, bn = False):
    if head_conv:
        model = ResNet8_more3D(heads, head_conv=head_conv, bn=bn)
    else:
        model = ResNet8_last3D(heads, head_conv = head_conv, bn=bn)
    return model