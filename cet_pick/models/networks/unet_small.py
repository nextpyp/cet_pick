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


class TomoUNet(nn.Module):
    def __init__(self, n_blocks, heads, head_conv,last_k = 3):
        self.heads = heads 
        self.n_blocks = n_blocks 
        super(TomoUNet, self).__init__()

        self.unet = UNet(1, out_channels = 32, n_blocks = self.n_blocks, planar_blocks = (), merge_mode = 'concat', dim=2)
        self.feature_head = nn.Sequential(nn.Conv3d(32, head_conv, kernel_size = (3,3,3), dilation = (1,1,1), padding=(1,1,1), bias=False), 
                    nn.ReLU(inplace=True),
                    nn.Conv3d(head_conv, head_conv, kernel_size = (3,last_k,last_k), dilation = (1,1,1), padding=(1,int((last_k-1)//2),int((last_k-1)//2)), bias=False), 
                    # nn.AvgPool3d(kernel_size=(3,7,7), padding=(1,3,3), stride=(1,1,1)),
                    # nn.Conv3d(64, 128, kernel_size = (1,1,1), dilation = (1,1,1), padding=(0,0,0), bias=False),
                    # nn.Conv3d(32, 32, kernel_size = (3,3,3), dilation = (1,1,1), padding=(1,1,1), bias=False),
                    nn.ReLU(inplace=True),
                    # nn.Conv3d(64, 64, kernel_size = (1,1,1), dilation = (1,1,1), padding=(0,0,0), bias=False),
                    # nn.ReLU(inplace=True),
            )

        fill_fc_weights(self.feature_head)

        for head in self.heads:
            classes = self.heads[head]

            fc = nn.Conv3d(head_conv, classes, kernel_size = (3,1,1), stride = 1, padding=(1,0,0), bias=False)
            # if 'hm' in head:
            #     fc.bias.data.fill_(-2.19)
            # else:
            fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        if len(x.shape) > 4:
            x = x.squeeze()
        b, d, h, w = x.shape
        if b > 1:
            x = x.reshape((-1,h,w)).contiguous()
            x = x.unsqueeze(1)
        else:
            x = x.permute(1,0,2,3)
        x = self.unet(x) 
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

    def init_weights(self, num_layers):
        if 1:
            url = model_urls['unet{}'.format(num_layers)]
            # pretrained_state_dict = model_zoo.load_url(url)
            pretrained_state_dict = torch.load(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
            print('=> loaded pretrained model {}'.format(url))
            # print('=> loading pretrained model {}'.format(url))
            # # self.load_state_dict(pretrained_state_dict, strict=False)
            # self._load_pretrained(pretrained_state_dict, inchans=1)
            # print('=> init deconv weights from normal distribution')
            # for name, m in self.deconv_layers.named_modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         nn.init.constant_(m.weight, 1)
            #         nn.init.constant_(m.bias, 0)


def get_tomo_unet_small(num_layers, heads, head_conv = 32,last_k=3):
    model = TomoUNet(num_layers, heads, head_conv, last_k)
    # model.init_weights(num_layers)
    return model



