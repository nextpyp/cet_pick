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
from cet_pick.models.networks.resnet_new import get_resnet_new
from cet_pick.models.networks.simsiam_model import get_simsiam_net_small
from cet_pick.models.networks.simsiam_model_3d import get_simsiam_net_small_3d
from cet_pick.models.networks.moco_encoder_3d import get_moco_net_small_3d
from cet_pick.models.networks.simsiam_model_2d import get_simsiam2d_net_small
from cet_pick.models.networks.simsiam_model_2d import get_clustering_net_small
from cet_pick.models.networks.simsiam_model_2d3d import get_simsiam2d3d_net_small
from cet_pick.models.networks.simsiam_model_2d3d import get_clustering2d3d_net_small
from cet_pick.models.networks.wideresnet import get_simsiam_wresnet
from cet_pick.models.networks.pyramidnet import get_simsiam_pyramidnet
from cet_pick.models.networks.simsiam_ressmall import get_simsiam_resnet_small
from cet_pick.models.networks.denoise_network import get_denoise_network
from cet_pick.models.networks.simsiam_model_2d import get_moco2d_net_small

_model_factory = {
    'res': get_tomo_net,
    'unet': get_tomo_unet_small,
    'class': get_tomo_class_net,
    'small': get_tomo_class_net_small,
    'ressmall': get_tomo_net_small,
    'p3d': get_tomo_p3d_net_small,
    'res3d': get_tomo_net_3d,
    'unetcla':get_tomo_unet_small_class,
    'resclass':get_resnet_new,
    'simsiam': get_simsiam_net_small,
    'simsiam3d': get_simsiam_net_small,
    'moco3d': get_moco_net_small_3d,
    'simsiam2d': get_simsiam2d_net_small,
    'simsiamwide3d': get_simsiam_wresnet,
    'simsiampyr3d':get_simsiam_pyramidnet,
    'simsiamsmall3d': get_simsiam_resnet_small,
    'scan2d': get_clustering_net_small,
    'simsiam2d3d': get_simsiam2d3d_net_small,
    # 'simsiam3d': get_simsiam2d_net_small,
    'scan2d3d': get_clustering2d3d_net_small,
    'denoise':  get_denoise_network,
    'moco2d': get_moco2d_net_small

}

def create_model_scan(arch, heads, head_conv, nclusters=3, nheads=1):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv, n_clusters = nclusters, nheads = nheads)
    return model 
    
def create_model(arch, heads, head_conv, last_k=0):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv,last_k = last_k)
    return model


def load_model_selflabel(model, model_path,optimizer = None, resume = False, lr = None, lr_step = None, model_only = False):
    start_epoch = 0 
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    all_heads = [k for k in state_dict_.keys() if 'cluster_head' in k]
    best_head_weight = state_dict_['cluster_head.%d.weight' %(checkpoint['best_loss_head'])]
    best_head_bias = state_dict['cluster_head.%d.bias' %(checkpoint['best_loss_head'])]
    pass


def load_model_scan(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None, model_only=False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    best_loss = checkpoint['best_loss']
    best_loss_head = checkpoint['best_loss_head']
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
        return model, optimizer, start_epoch, best_loss, best_loss_head
    else:
        return model

def load_pretrain_scan(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None, model_only = False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict['backbone.'+k[7:]] = state_dict_[k]
        else:
            state_dict['backbone.'+k] = state_dict_[k]

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
        # print('state dict k', k)
        # print('backbone.', 'backbone.'+k)

        if k in model_state_dict:
            # print('model state dict k', k)
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


def save_model_scan(path, epoch, model, optimizer=None, best_loss=None, best_loss_head =None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
          'state_dict': state_dict}

    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    if not (best_loss is None):
        data['best_loss'] = best_loss 

    if not (best_loss_head is None):
        data['best_loss_head'] = best_loss_head

    torch.save(data, path)

def save_model(path, epoch, model, optimizer=None, task=None, **kwargs):
    # if task == 'scan':

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
          'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    if task == 'scan':
        data['head'] = kwargs['head']
    torch.save(data, path)
