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
from cet_pick.models.networks.unet import UNet 
import numpy as np 

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    'resnet8': 'https://github.com/tbepler/topaz/blob/master/topaz/pretrained/detector/resnet8_u64.sav'
}


model_urls_http = {
    "resnet18": "http://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "http://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "http://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "http://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "http://download.pytorch.org/models/resnet152-394f9c45.pth",
    'resnet8': 'http://github.com/tbepler/topaz/blob/master/topaz/pretrained/detector/resnet8_u64.sav'
}
# 
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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class PreUNet(nn.Module):
    def __init__(self, n_blocks, heads, head_conv):
        super(PreUNet, self).__init__()
        self.heads = heads 
        self.n_blocks = n_blocks
        self.features = UNet(1, out_channels = 128, n_blocks = self.n_blocks, planar_blocks = (), merge_mode = 'concat', dim=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, 128)
        fill_fc_weights(self.fc)
        for head in self.heads:
            classes = self.heads[head]
            if 'proj' in head:
                fc = nn.Sequential(nn.Linear(128, 128, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(128, 128, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(128, 128, bias=False),
                                   nn.BatchNorm1d(128, affine=False))
            if 'pred' in head:
                fc = nn.Sequential(nn.Linear(128, 128, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(128, 128)
                    )
            fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward_test(self, x1):
        if len(x1.shape) > 4:
            x1 = x1.squeeze(dim=1)
        b, c, h, w = x1.shape
        x1 = self.features(x1)
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
        b, c, h, w = x1.shape
        x1 = self.features(x1)
        x2 = self.features(x2)
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


class ResNet(nn.Module):
    def __init__(self, heads, *args, **kwargs):
        super(ResNet, self).__init__()

        if 'pooling' in kwargs:
            pooling = kwargs['pooling']
            if pooling == 'max':
                kwargs['pooling'] = MaxPool

        modules = self.make_modules(**kwargs)
        self.heads = heads
        self.features = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = 64
    
        self.fc = nn.Linear(self.latent_dim, self.out_dim)

        fill_fc_weights(self.fc)
        for head in self.heads:
            classes = self.heads[head]
            if 'proj' in head:
                fc = nn.Sequential(nn.Linear(self.out_dim, self.out_dim, bias=False),
                                   nn.BatchNorm1d(self.out_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.out_dim, self.out_dim, bias=False),
                                   nn.BatchNorm1d(self.out_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.out_dim, self.out_dim, bias=False),
                                   nn.BatchNorm1d(self.out_dim, affine=False))
            if 'pred' in head:
                fc = nn.Sequential(nn.Linear(self.out_dim, self.out_dim, bias=False),
                                   nn.BatchNorm1d(self.out_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.out_dim, self.out_dim)
                    )
            fill_fc_weights(fc)
            self.__setattr__(head, fc)


    def forward_test(self, x1):
        if len(x1.shape) > 4:
            x1 = x1.squeeze(dim=1)
        b, c, h, w = x1.shape
        x1 = self.features(x1)
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
        b, c, h, w = x1.shape
        x1 = self.features(x1)
        x2 = self.features(x2)
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
    def _load_pretrained(self, checkpoint):
        
        model_state_dict = self.state_dict()
        state_dict_ = checkpoint
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[16:]] = state_dict_[k]
            else:
                state_dict[k[9:]] = state_dict_[k]

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

class ResNet6(ResNet):
    def make_modules(self, units=[32, 64, 128], bn=True, dropout=0.0, activation=nn.ReLU, **kwargs):
        if units is None:
            units = [32, 64, 128]
        elif type(units) is not list:
            units = int(units)
            units = [units, 2*units, 4*units]

        self.num_features = units[-1]

        modules = [
                BasicConv2d(1, units[0], 5, bn=bn, activation=activation),
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[0], units[0], units[1], dilation=1, bn=bn, activation=activation),
                ]
        # modules.append(MaxPool(3, stride=2))
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
            self.stride = 1
        stride = self.stride

        modules = [
                BasicConv2d(1, units[0], 7, stride=stride, bn=bn, activation=activation),
                ]
        if pooling is not None:
            modules.append(pooling(3, stride=1))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[0], units[0], units[0], dilation=1, bn=bn, activation=activation),
                ResidA(units[0], units[0], units[1], dilation=1
                      , stride=1, bn=bn, activation=activation),
                ]
        if pooling is not None:
            modules.append(pooling(3, stride=2))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        modules += [
                ResidA(units[1], units[1], units[1], dilation=1, bn=bn, activation=activation),
                BasicConv2d(units[1], units[2], 5, bn=bn, activation=activation)
                ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout)) #, inplace=True))

        self.latent_dim = units[-1]

        return modules

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.og_stride = stride
        self.dilation = 1
        self.padding = 0

    def set_padding(self, pad):
        if pad:
            p = self.dilation*(self.kernel_size//2) # this is bugged in pytorch...
            #p = self.kernel_size//2
            self.pool.padding = (p, p)
            self.padding = p
        else:
            self.pool.padding = (0,0)
            self.padding = 0

    def fill(self, stride):
        self.pool.dilation = stride
        self.pool.padding = self.pool.padding*stride
        self.pool.stride = 1
        self.dilation = stride
        self.stride = 1
        return self.og_stride

    def unfill(self):
        self.pool.dilation = 1
        self.pool.padding = self.pool.padding//self.dilation
        self.pool.stride = self.og_stride
        self.dilation = 1
        self.stride = self.og_stride

    def forward(self, x):
        return self.pool(x)

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


    def forward(self, x):
        y = self.conv(x)
        if hasattr(self, 'bn'):
            y = self.bn(y)
        return self.act(y)

class ResidA(nn.Module):
    def __init__(self, nin, nhidden, nout, dilation=1, stride=1
                , activation=nn.ReLU, bn=False):
        super(ResidA, self).__init__()

        self.bn = bn
        bias = not bn

        if nin != nout:
            self.proj = nn.Conv2d(nin, nout, 1, stride=stride, bias=False)
        
        self.conv0 = nn.Conv2d(nin, nhidden, 3, bias=bias, padding = 1)
        if self.bn:
            self.bn0 = nn.BatchNorm2d(nhidden)
        self.act0 = activation(inplace=True)

        self.conv1 = nn.Conv2d(nhidden, nout, 3, dilation=dilation, stride=stride
                              , bias=bias, padding = 1)
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


    def forward(self, x):
        h = self.conv0(x)
        if self.bn:
            h = self.bn0(h)
        h = self.act0(h)

        y = self.conv1(h)
        # edge = self.conv0.dilation[0] + self.conv1.dilation[0]
        # x = x[:,:,edge:-edge,edge:-edge]
        if hasattr(self, 'proj'):
            x = self.proj(x)
        elif self.conv1.stride[0] > 1:
            x = x[:,:,::self.stride,::self.stride]
        y = y + x
        if self.bn:
            y = self.bn1(y)
        y = self.act1(y)

        return y

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



def gaussian_mask(c,d,h,w,sigma):
    b = np.arange(-d//2,d//2)
    hh = np.exp(-b*b/(2*sigma*sigma))
    m = np.ones((c,d,h,w))
    for i, j in enumerate(hh):
        m[:,i,:,:] = m[:,i,:,:] * j
    return torch.from_numpy(m.astype(np.float32))

class TomoResClassifier2D3D(nn.Module):
    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads 
        self.deconv_with_bias = False 

        super(TomoResClassifier2D3D, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.deconv_layers = self._make_deconv_layer(
        #     4,
        #     [256, 128, 64, 32],
        #     [4, 4, 4, 4],
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = head_conv
        self.fc = nn.Linear(512*block.expansion, self.out_dim)
        fill_fc_weights(self.fc)
        # self.layer_3d = nn.Conv3d(256*block.expansion, 256*block.expansion, kernel_size=3, padding=1, bias=False)
        # self.bn3d = nn.BatchNorm3d(256*block.expansion, momentum=BN_MOMENTUM)
        # self.relu3d = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv1d(1, 1, kernel_size = 3, padding = 1, bias=False)
        for head in self.heads:
            classes = self.heads[head]
            if 'proj' in head:
                fc = nn.Sequential(nn.Linear(self.out_dim, self.out_dim, bias=False),
                                   nn.BatchNorm1d(self.out_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.out_dim, self.out_dim, bias=False),
                                   nn.BatchNorm1d(self.out_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.out_dim, self.out_dim, bias=False),
                                   nn.BatchNorm1d(self.out_dim, affine=False))
            if 'pred' in head:
                fc = nn.Sequential(nn.Linear(self.out_dim, self.out_dim, bias=False),
                                   nn.BatchNorm1d(self.out_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.out_dim, self.out_dim)
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


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

    def forward_test(self, x1_2d, x1_3d):
        if len(x1_2d.shape) > 4:
            x1_2d = x1_2d.squeeze(dim=1)
        b, c, h, w = x1_2d.shape
        b2, c2, h2, w2 = x1_3d.shape
        x1 = torch.cat([x1_2d, x1_3d], dim = 0)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        # x1 = self.maxpool(x1)
# 
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        # x1 = self.layer4(x1)
        dd, ch, hh, ww = x1.shape
        x1 = self.avgpool(x1)
        x1 = x1.reshape(x1.size(0), -1)
        x1_all = torch.chunk(x1, 2, dim=0)
        x1 = torch.cat(x1_all, dim = 1)
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

    def forward(self,x1_2d, x1_3d, x2_2d, x2_3d):
        #input is 1 * 64 * 512 * 512 
        #reshape it to 64 * 1 * 512 * 512 to treat each slice individually
        if len(x1_2d.shape) > 4:
            x1_2d = x1_2d.squeeze(dim=1)
        if len(x2_2d.shape) > 4:
            x2_2d = x2_2d.squeeze(dim=1)
        b, c, h, w = x1_2d.shape
        b2, c2, h2, w2 = x1_3d.shape
        x1 = torch.cat([x1_2d, x1_3d], dim = 0)
        x2 = torch.cat([x2_2d, x2_3d], dim = 0)

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        # x2 = self.maxpool(x2)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        # x2 = self.layer4(x2)
        dd, ch, hh, ww = x1.shape
        x1 = self.avgpool(x1)
       
        x1 = x1.reshape(x1.size(0), -1)
        x1_all = torch.chunk(x1, 2, dim=0)
        x1 = torch.cat(x1_all, dim = 1)
        x1 = self.fc(x1)
        x2 = self.avgpool(x2)
        x2 = x2.reshape(x2.size(0), -1)
        x2_all = torch.chunk(x2, 2, dim=0)
        x2 = torch.cat(x2_all, dim = 1)
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
        try:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            self._load_pretrained(pretrained_state_dict, inchans=1)
        except:
            print('https url not working, trying http based url....')
            url = model_urls_http['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            self._load_pretrained(pretrained_state_dict, inchans=1)

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone 
        self.nheads = nheads 
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(256, nclusters) for _ in range(self.nheads)])

    def forward(self, x_2d, x_3d, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone.forward_test(x_2d, x_3d)
            features_proj = features['proj']
            out = [cluster_head(features_proj) for cluster_head in self.cluster_head]
        elif forward_pass == 'backbone':
            features = self.backbone.forward_test(x_2d, x_3d)
            out = features['proj']

        elif forward_pass == 'head':
            out = [cluster_head(x_2d) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone.forward_test(x_2d, x_3d)
            features_proj = features['proj']
            cluster_out = [cluster_head(features_proj) for cluster_head in self.cluster_head]
            out = {'features': features_proj, 'output': cluster_out}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out 
        
def get_clustering2d3d_net_small(num_layers, heads, head_conv=32, n_clusters=3, nheads=1):
    block_class, layers = resnet_spec[num_layers]
    backbone = TomoResClassifier2D3D(block_class, layers, heads, head_conv=0)
    # backbone = PreUNet(num_layers, heads, head_conv=16)
    backbone.init_weights(num_layers)
    model = ClusteringModel(backbone, nclusters=n_clusters, nheads=nheads)
    return model

def get_simsiam2d3d_net_small(num_layers, heads, head_conv = 32, last_k = 0):
    block_class, layers = resnet_spec[num_layers]
    model = TomoResClassifier2D3D(block_class, layers, heads, head_conv)

    model.init_weights(num_layers)
    return model


