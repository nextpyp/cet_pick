# coding=utf-8

"""
ResNet.

Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385

    [2] https://github.com/pytorch/vision.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import torch
import torch.nn as nn
import logging
from torch.autograd import Variable

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

BN_MOMENTUM = 0.1
__all__ = ['pyramidnet272', 'pyramidnet164']

_inplace_flag = True

class ShakeDropFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        if training:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(0, 1)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            # beta = beta.reshape(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None


class ShakeDrop(nn.Module):

    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_shakedrop=1.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=_inplace_flag)
        self.downsample = downsample
        self.stride = stride
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                                        featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_shakedrop=1.0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, (planes * 1), kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d((planes * 1))
        self.conv3 = nn.Conv2d((planes * 1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=_inplace_flag)
        self.downsample = downsample
        self.stride = stride
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                                            featuremap_size[1]).fill_(0))
            # for summary parameters and FLOPs
            #padding = torch.autograd.Variable(
            #           torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
            #                           featuremap_size[1]).type_as(shortcut))
            # out += torch.cat((shortcut, padding), 1)
            out = out + torch.cat((shortcut, padding), 1)
        else:
            # out += shortcut
            out = out + shortcut
        return out

class PyramidTomoNet(nn.Module):
    def __init__(self, heads, bottleneck=True, depth=272, alpha=200, split_factor=1):
        super(PyramidTomoNet, self).__init__()
        self.inplanes=16   
        if bottleneck:
                n = int((depth - 2) / 9)
                block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock
        self.heads = heads
        self.addrate = alpha / (3 * n * (split_factor ** 0.5))
        self.final_shake_p = 0.5 / (split_factor ** 0.5)
        print('INFO:PyTorch: PyramidNet: The add rate is {}, '
            'the final shake p is {}'.format(self.addrate, self.final_shake_p))

        self.ps_shakedrop = [1. - (1.0 - (self.final_shake_p / (3 * n)) * (i + 1)) for i in range(3 * n)]

        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d(1, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim
        self.layer1 = self.pyramidal_make_layer(block, n)
        self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
        self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=_inplace_flag)

        self.feature_3d = nn.Sequential(nn.Conv3d(self.final_featuremap_dim, self.final_featuremap_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.final_featuremap_dim, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
            )

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(self.final_featuremap_dim, 128)
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
            self.__setattr__(head, fc)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        assert len(self.ps_shakedrop) == 0, self.ps_shakedrop

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:  # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample,
                            p_shakedrop=self.ps_shakedrop.pop(0)))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(
                block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1,
                        p_shakedrop=self.ps_shakedrop.pop(0)))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

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

    def _load_pretrained(self, checkpoint, inchans=1):
        state_dict ={}

        state_dict_ = checkpoint
        for k in state_dict_:
            print('k', k)
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[16:]] = state_dict_[k]
            else:
                state_dict[k[9:]] = state_dict_[k]
            # print('new state dict', state_dict.keys())

        msg = 'If you see this, your model does not fully load the ' + \
            'pre-trained weight. Please make sure ' + \
            'you have correctly specified --arch xxx ' + \
            'or set the correct --num_classes for your own dataset.'
        print('state_dict', state_dict.keys())
        if inchans == 1:
            conv1_weights = state_dict['conv1.weight']
            print('conv1_weights', conv1_weights.shape)
            state_dict['conv1.weight'] = conv1_weights.sum(dim=1, keepdim=True)
        elif inchans != 3:
            assert False, "Invalid number of in channels"
        model_state_dict = self.state_dict()
        # model_state_dict = model.state_dict()
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
        # print('batch size', b)
        if b > 1:
            x1 = x1.reshape((-1,h,w)).contiguous()
            x1 = x1.unsqueeze(1)
            x2 = x2.reshape((-1,h,w)).contiguous()
            x2 = x2.unsqueeze(1)
        else:
            x1 = x1.permute(1,0,2,3)
            x2 = x2.permute(1,0,2,3)
        # x1 = self.layer0(x1)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.bn_final(x1)
        x1 = self.relu_final(x1)
       
        # x2 = self.layer0(x2)
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.bn_final(x2)
        x2 = self.relu_final(x2)
        # if self.layer4 is not None:
        #     x2 = self.layer4(x2)
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

            # ret[head] = self.__getattr__(head)(x)
            # print('head', head)
            # print('out', ret[head].shape)

        return [ret1, ret2]

def get_simsiam_pyramidnet(num_layers, heads, head_conv,**kwargs):
    model = PyramidTomoNet(heads, bottleneck=True, depth=272, alpha=200)
    path = '/nfs/bartesaghilab/qh36/3D_picking/cet_pick_github/cet_pick/cet_pick/models/pretrain/pyramid_model_best.pth.tar'
    state_dict_model = torch.load(path)
    checkpoint = state_dict_model['state_dict']
    model._load_pretrained(checkpoint)
    del state_dict_model
    return model

# heads = {'proj': 128, 'pred': 128}
# 
# print('pyramidnet', pyramidnet)
# path = '/nfs/bartesaghilab/qh36/3D_picking/cet_pick_github/cet_pick/cet_pick/models/pretrain/pyramid_model_best.pth.tar'
# state_dict_model = torch.load(path)
# checkpoint = state_dict_model['state_dict']
# it = 0
# for k in checkpoint.keys():
#     param = checkpoint[k]
#     if it < 10:
#         print(k, param.shape)
#     it += 1
# pyramidnet._load_pretrained(checkpoint)
