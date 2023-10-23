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
    def __init__(self, n_blocks, heads, head_conv):
        self.heads = heads 
        self.n_blocks = n_blocks 
        super(TomoUNet, self).__init__()

        self.unet = UNet(1, out_channels = 32, n_blocks = self.n_blocks, planar_blocks = (), merge_mode = 'concat', dim=2)
        self.feature_head = nn.Sequential(nn.Conv3d(32, 16, kernel_size = (3,3,3), padding=(1,1,1), bias=True), 
                    nn.ReLU(inplace=True)
            )

        fill_fc_weights(self.feature_head)

        for head in self.heads:
            classes = self.heads[head]

            fc = nn.Conv3d(16, classes, kernel_size = 1, stride = 1, padding=0, bias=True)
            if 'hm' in head:
                fc.bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        # print('input', x.shape)
        # print(len(x.shape))
        if len(x.shape) > 4:
            x = x.squeeze()
        b, d, h, w = x.shape
        # print('batch size', b)
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
        # print('x final', x.shape)
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
                # print('outt_feature', out.shape)
                # print('after flatten', out.shape)
            # print('out_feature', out_f.shape)
            # out = self.__getattr__(head_out)(out_f)
            # print('out_final', out.shape)
            # out = out_conv(out)
            # out = out.permute(1,0,2,3)
            # out = out.unsqueeze(0)
            ret[head] = out
            # ret[head_out] = out_f
            # print('head', head)
            # print('out', ret[head].shape)

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


def get_tomo_unet_small(num_layers, heads, head_conv=16):
    model = TomoUNet(num_layers, heads, head_conv=16)
    # model.init_weights(num_layers)
    return model



