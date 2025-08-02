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


class TomoResNet(nn.Module):
    def __init__(self, block, layers, heads, head_conv, last_k):
        self.inplanes = 64
        self.heads = heads 
        self.deconv_with_bias = False 

        super(TomoResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.deconv_layers = self._make_deconv_layer(
            2,
            [64, 32],
            [4, 4],
        )

        self.feature_head = nn.Sequential(nn.Conv3d(32, head_conv, kernel_size = (3,last_k,last_k), padding=(1,int((last_k-1)//2),int((last_k-1)//2)), bias=True), 
                    nn.ReLU(inplace=True)
            )
        fill_fc_weights(self.feature_head)
        # self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)
        # self.conv3 = nn.Conv1d(1, 1, kernel_size = 3, padding = 1, bias=False)
        # self.feature_head = nn.Sequential()
        for head in self.heads:
            # print('head', head)
            # head_out = head + '_out'
            classes = self.heads[head]
            # if head_conv > 0:
            #     fc = nn.Sequential(
            #         nn.Conv3d(32, head_conv, kernel_size = 3, padding=1, bias=True), 
            #         nn.ReLU(inplace=True)
            #         )
            #     fl = nn.Conv3d(head_conv, classes, kernel_size = 1, stride=1, padding=0, bias=True)
            #     if 'hm' in head:
            #         fl.bias.data.fill_(-2.19)
            #     # else:
            #     fill_fc_weights(fc)
            # else:
            # this can be viewed as projection head in contrastive learning
            fc = nn.Conv3d(head_conv, classes, kernel_size = 1, stride = 1, padding=0, bias=True)
            if 'hm' in head:
                fc.bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
            # self.__setattr__(head_out, fl)
            # self.__setattr__()




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
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self,x):
        #input is 1 * 64 * 512 * 512 
        #reshape it to 64 * 1 * 512 * 512 to treat each slice individually
        if len(x.shape) > 4:
            x = x.squeeze()
        b, d, h, w = x.shape
        # print('batch size', b)
        if b > 1:
            x = x.reshape((-1,h,w)).contiguous()
            x = x.unsqueeze(1)
        else:
            x = x.permute(1,0,2,3)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.deconv_layers(x)
        # now the output shape should be 64 * 64 * 128 * 128
        # we do a covolution so it becomes 64 * 1 * 128 * 128
        dd, ch, hh, ww = x.shape
        if b > 1:
            x = x.reshape((b, d, ch, hh, ww)).contiguous()
            x = x.permute(0,2,1,3,4)
        else:
            x = x.permute(1,0,2,3)
            x = x.unsqueeze(0)
        x = self.feature_head(x)
        ret = {}
        for head in self.heads:
            out = self.__getattr__(head)(x)
            if 'proj' in head:
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
        try:
            url = model_urls['resnet{}'.format(num_layers)]
            local_url = os.path.join('/opt/pyp/external/models', os.path.basename(url))
            pretrained_state_dict = torch.load(local_url)
            print('=> loading pretrained model {}'.format(local_url))
            self._load_pretrained(pretrained_state_dict, inchans=1)
        except:
            print(f'Could not load pretrained model {url}')
            try:
                pretrained_state_dict = model_zoo.load_url(url)
                # url  = '/nfs/bartesaghilab/qh36/3D_picking/cet_pick/cet_pick/exp/tcla/classify_new/model_last.pth'
                print('=> loading pretrained model {}'.format(url))
                # self.load_state_dict(pretrained_state_dict, strict=False)
                # print('=> loading pretrained model {}'.format(url))
                # # self.load_state_dict(pretrained_state_dict, strict=False)
                self._load_pretrained(pretrained_state_dict, inchans=1)
            except:
                raise ValueError(f'Could not load pretrained model {url}')
        print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_tomo_net_small(num_layers, heads, head_conv = 32, last_k = 3):
    block_class, layers = resnet_spec[num_layers]

    model = TomoResNet(block_class, layers, heads, head_conv, last_k)
    model.init_weights(num_layers)
    return model


