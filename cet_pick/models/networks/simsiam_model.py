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
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.001)

def gaussian_mask(c,d,h,w,sigma):
    b = np.arange(-d//2,d//2)
    hh = np.exp(-b*b/(2*sigma*sigma))
    m = np.ones((c,d,h,w))
    for i, j in enumerate(hh):
        m[:,i,:,:] = m[:,i,:,:] * j
    return torch.from_numpy(m.astype(np.float32))

# class TomoSmallResClassifier(nn.Module):
#     def __init__(self, block, layers, heads, head_conv):
#         self.inplanes = 16
#         self.heads = heads 
#         super(TomoSmallResClassifier, self).__init__()
        

class TomoResClassifier(nn.Module):
    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads 
        self.deconv_with_bias = False 

        super(TomoResClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.deconv_layers = self._make_deconv_layer(
        #     4,
        #     [256, 128, 64, 32],
        #     [4, 4, 4, 4],
        # )
        # self.gaussian_mask = gaussian_mask(256,10,2,2,2)
        # print('gaussian_mask', self.gaussian_mask.get_device())
        # self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)
        self.feature_3d = nn.Sequential(nn.Conv3d(256*block.expansion, 256*block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256*block.expansion, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
            )
        fill_fc_weights(self.feature_3d)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(256*block.expansion, 256)
        fill_fc_weights(self.fc)
        # self.layer_3d = nn.Conv3d(256*block.expansion, 256*block.expansion, kernel_size=3, padding=1, bias=False)
        # self.bn3d = nn.BatchNorm3d(256*block.expansion, momentum=BN_MOMENTUM)
        # self.relu3d = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv1d(1, 1, kernel_size = 3, padding = 1, bias=False)
        for head in self.heads:
            classes = self.heads[head]
            if 'proj' in head:
                fc = nn.Sequential(nn.Linear(256, 256, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(256, 256, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(256, 256, bias=False),
                                   nn.BatchNorm1d(256, affine=False))
            if 'pred' in head:
                fc = nn.Sequential(nn.Linear(256, 256, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(256, 256)
                    )
            fill_fc_weights(fc)

            # if head_conv > 0:
            #     fc = nn.Sequential(
            #         nn.Conv3d(32, head_conv, kernel_size = 3, padding=1, bias=True), 
            #         nn.ReLU(inplace=True),
            #         nn.Conv3d(head_conv, classes, kernel_size = 1, stride=1, padding=0, bias=True))
            #     if 'hm' in head:
            #         fc[-1].bias.data.fill_(-2.19)
            #     else:
            #         fill_fc_weights(fc)
            # else:
            #     # fc = nn.Conv3d(1, classes, kernel_size = 1, stride = 1, padding=0, bias=True)
            #     fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            #         nn.Flatten(),
            #         nn.Linear(128, 64),
            #         nn.ReLU(),
            #         nn.Dropout(0.5),
            #         nn.Linear(64, 1)
            #         )
            #     if 'hm' in head:
            #         fc[-1].bias.data.fill_(-2.19)
            #     else:
            #         fill_fc_weights(fc)
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
                # nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
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

    def forward_test(self, x1):
        if len(x1.shape) > 4:
            x1 = x1.squeeze(dim=1)
        # self.gaussian_mask = self.gaussian_mask.to(x1.get_device())
        # print()
        b, d, h, w = x1.shape
        # print('batch size', b)
        if b > 1:
            x1 = x1.reshape((-1,h,w)).contiguous()
            x1 = x1.unsqueeze(1)
        else:
            x1 = x1.permute(1,0,2,3)
        
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        dd, ch, hh, ww = x1.shape
        if b > 1:
            x1 = x1.reshape((b, d, ch, hh, ww)).contiguous()
            x1 = x1.permute(0,2,1,3,4)
        else:
            x1 = x1.permute(1,0,2,3)
            x1 = x1.unsqueeze(0)
        x1 = self.feature_3d(x1)
        # print('before pool', x1.shape)
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
        #input is 1 * 64 * 512 * 512 
        #reshape it to 64 * 1 * 512 * 512 to treat each slice individually
        # print('x1', x1.get_device())
        # self.gaussian_mask = self.gaussian_mask.to(x1.get_device())
        if len(x1.shape) > 4:
            x1 = x1.squeeze(dim=1)
        if len(x2.shape) > 4:
            x2 = x2.squeeze(dim=1)
        b, d, h, w = x1.shape
        if b > 1:
            x1 = x1.reshape((-1,h,w)).contiguous()
            x1 = x1.unsqueeze(1)
            x2 = x2.reshape((-1,h,w)).contiguous()
            x2 = x2.unsqueeze(1)
        else:
            x1 = x1.permute(1,0,2,3)
            x2 = x2.permute(1,0,2,3)
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
        dd, ch, hh, ww = x1.shape
        if b > 1:
            x1 = x1.reshape((b, d, ch, hh, ww)).contiguous()
            x1 = x1.permute(0,2,1,3,4)
            x2 = x2.reshape((b, d, ch, hh, ww)).contiguous()
            x2 = x2.permute(0,2,1,3,4)
        else:
            x1 = x1.permute(1,0,2,3)
            x1 = x1.unsqueeze(0)
            x2 = x2.permute(1,0,2,3)
            x2 = x2.unsqueeze(0)
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



        return [ret1, ret2]


    def _load_pretrained(self, state_dict, inchans=3):
        if inchans == 1:
            conv1_weights = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weights.sum(dim=1, keepdim=True)
        elif inchans != 3:
            assert False, "Invalid number of in channels"

        model_state_dict = self.state_dict()
        msg = 'If you see this, your model does not fully load the ' + \
            'pre-trained weight. Please make sure ' + \
            'you have correctly specified --arch xxx ' + \
            'or set the correct --num_classes for your own dataset.'
        for k in state_dict:
            # print('k', k)
            # if not k in model_state_dict:
                # print('diff', k)
        # for k in model_state_dict:

            # print('model', k)
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
        # print('state_dict', self.state_dict())
        # for k in state_dict:
        #     if 'fc' in k:
        #         print(state_dict[k])
        # self.load_state_dict(state_dict, strict=False)
    def init_weights(self, num_layers, local_path = None):
        if local_path is None:
            try:
                url = model_urls['resnet{}'.format(num_layers)]
                local_url = os.path.join('/opt/pyp/external/models', os.path.basename(url))
                pretrained_state_dict = torch.load(local_url)
                print('=> loading pretrained model {}'.format(local_url))
                self._load_pretrained(pretrained_state_dict, inchans=1)                          
            except:
                print(f'Could not load pretrained model {local_url}')
                try:
                    pretrained_state_dict = model_zoo.load_url(url)
                    print('=> loading pretrained model {}'.format(url))
                    # self.load_state_dict(pretrained_state_dict, strict=False)
                    self._load_pretrained(pretrained_state_dict, inchans=1)
                except:
                    print(f'Could not load pretrained model {url}')
                    try:
                        url = model_urls_http['resnet{}'.format(num_layers)]
                        pretrained_state_dict = model_zoo.load_url(url)
                        print('=> loading pretrained model {}'.format(url))
                        # self.load_state_dict(pretrained_state_dict, strict=False)
                        self._load_pretrained(pretrained_state_dict, inchans=1)
                    except:
                        raise ValueError(f'Could not load pretrained model {url}')
        else:
            url  = local_path
            pretrained_state_dict = torch.load(url)
            print('=> loading pretrained model {}'.format(url))
            self._load_pretrained(pretrained_state_dict, inchans=1)

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_simsiam_net_small(num_layers, heads, head_conv = 32, last_k = 0, local_path = None):
    block_class, layers = resnet_spec[num_layers]

    # model = TomoResClassifier
    model = TomoResClassifier(block_class, layers, heads, head_conv=0)
    model.init_weights(num_layers, local_path = local_path)
    return model


