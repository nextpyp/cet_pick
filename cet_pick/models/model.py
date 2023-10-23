from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from cet_pick.models.networks.resnet import get_tomo_net
from cet_pick.models.networks.unet_small import get_tomo_unet_small
from cet_pick.models.networks.classifier import get_tomo_class_net
from cet_pick.models.networks.classifier_small import get_tomo_class_net_small
from cet_pick.models.networks.resnet_small import get_tomo_net_small
from cet_pick.models.networks.p3d_small import get_tomo_p3d_net_small
from cet_pick.models.networks.resnet_3d_small import get_tomo_net_3d
from cet_pick.models.networks.unet_small_class import get_tomo_unet_small_class

_model_factory = {
    'res': get_tomo_net,
    'unet': get_tomo_unet_small,
    'class': get_tomo_class_net,
    'small': get_tomo_class_net_small,
    'ressmall': get_tomo_net_small,
    'p3d': get_tomo_p3d_net_small,
    'res3d': get_tomo_net_3d,
    'unetcla':get_tomo_unet_small_class
}

def create_model(arch, heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model

# def load_model_only(mode, model_path):

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None, model_only = False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
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
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
        for step in lr_step:
            if start_epoch >= step:
                start_lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr
        print('Resumed optimizer with start lr', start_lr)
    else:
        print('No optimizer parameters in checkpoint.')
    if optimizer is not None and not model_only:
        return model, optimizer, start_epoch
    else:
        return model

def save_center(path, epoch, center):
    data = {'epoch': epoch, 'center': center}
    torch.save(data, path)

def load_center(center_path):
    ckpt = torch.load(center_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(center_path, ckpt['epoch']))
    center = ckpt['center']
    return center

def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
          'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
