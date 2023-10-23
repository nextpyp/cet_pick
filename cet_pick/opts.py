from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('task', default='tomo',
                             help='particle detection for cryoET/tomo datasets')
    self.parser.add_argument('--dataset', default='tomo',
                             help='cryoET dataset has default tomo')
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk')
    self.parser.add_argument('--demo', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 

    # system
    self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet
    #DDP configs:
    self.parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    self.parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    self.parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    self.parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    self.parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    
    # model
    self.parser.add_argument('--arch', default='res_18', 
                             help='model architecture. Currently tested'
                                  'ressmall_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | hourglass')
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '64 for resnets and 256 for dla.')
    self.parser.add_argument('--down_ratio', type=int, default=2,
                             help='output stride. Currently only supports 2.')

    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')
    
    # train
    self.parser.add_argument('--lr', type=float, default=1e-3, 
                             help='learning rate for batch size 1.')
    self.parser.add_argument('--lr_step', type=str, default='200, 400, 600',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=140,
                             help='total training epochs.')
    self.parser.add_argument('--lr_decay_rate', type=float, default=0.1)

    self.parser.add_argument('--cosine', action='store_true', help='use cosine annealing')

    self.parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')

    self.parser.add_argument('--batch_size', type=int, default=1,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')
    self.parser.add_argument('--bbox', type=int, default=32, help='bbox size')
    self.parser.add_argument('--cr_weight', type=float, default=0.1, help='weight for contrastive loss')
    self.parser.add_argument('--thresh', type=float, default=0.5, help='threshold for pos and neg contrastive')
    self.parser.add_argument('--temp', type=float, default=0.07, help='temperature for info nce loss')
    self.parser.add_argument('--tau', type=float, default=0.1, help='class prior probability')

    # test

    self.parser.add_argument('--nms', action='store_true',
                             help='run nms in testing.')
    self.parser.add_argument('--K', type=int, default=200,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_res', action='store_true',
                             help='fix testing resolution or keep '
                                  'the original resolution')
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')

    self.parser.add_argument('--out_thresh', type=float, default=0.25, help='output confidence threshold for particles')

    self.parser.add_argument('--with_score', action='store_true', help='whether to have score column in output text files')

    self.parser.add_argument('--contrastive', action='store_true',
                             help='whether contrastive training')

    self.parser.add_argument('--pn', action='store_true',
                             help='whether to use positive negative learning with fully labeled data')
    self.parser.add_argument('--ge', action='store_true',
                             help='whether to use generalized criteria loss')

    ### data 

    self.parser.add_argument('--train_img_txt', type=str, default='train_images.txt', help='path to file of training images')
    self.parser.add_argument('--train_coord_txt', type=str, default='train_coords.txt', help='path to file of training coords')
    self.parser.add_argument('--val_img_txt', type=str, default='val_images.txt', help='path to file of validation images')
    self.parser.add_argument('--val_coord_txt', type=str, default='val_coords.txt', help='path to file of validation coordinates')
    self.parser.add_argument('--test_img_txt', type=str, default = 'test_images.txt', help='path to file of test images')
    self.parser.add_argument('--test_coord_txt', type=str, default='test_coords.txt', help='path to file of test coords')

  

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

    opt.fix_res = not opt.keep_res
    # print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 16
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.warm:
      opt.warmup_from = 0.01 
      opt.warm_epochs = 10 
      if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        opt.warmup_to =  eta_min + (opt.lr - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.num_epochs)) / 2
      else:
        opt.warmup_to = opt.lr
        
    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]
      opt.master_batch_size = -1

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)
    # need opt root dir
    # opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.root_dir = os.getcwd()
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    opt.out_path = os.path.join(opt.save_dir, 'output_test')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.num_classes = dataset.num_classes

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    
    
    if opt.task == 'tomo':
      opt.heads = {'hm': opt.num_classes, 'proj': 16}
    elif opt.task == 'cr' or opt.task == 'semi' or opt.task =='semi3d':
      opt.heads = {'hm': opt.num_classes, 'proj':16}
    elif opt.task == 'fs':
      opt.heads = {'proj':16}
    elif opt.task == 'tcla':
      opt.heads ={'class':1}
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    default_dataset_info = {
      # 'ctdet': {'default_resolution': [512, 512], 'num_classes': 80, 
      #           'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
      #           'dataset': 'coco'},
      # 'exdet': {'default_resolution': [512, 512], 'num_classes': 80, 
      #           'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
      #           'dataset': 'coco'},
      # 'multi_pose': {
      #   'default_resolution': [512, 512], 'num_classes': 1, 
      #   'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
      #   'dataset': 'coco_hp', 'num_joints': 17,
      #   'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
      #                [11, 12], [13, 14], [15, 16]]},
      # 'ddd': {'default_resolution': [384, 1280], 'num_classes': 3, 
      #           'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
      #           'dataset': 'kitti'},
      'tomo': {'default_resolution': [512, 512], 'num_classes': 1, 'dataset': 'tomo'},
      'cr':{'default_resolution':[64, 64], 'num_classes':1, 'dataset': 'cr'},
      'semi': {'default_resolution':[64, 64], 'num_classes':1, 'dataset': 'semi'},
      'semi3d': {'default_resolution':[64, 64], 'num_classes':1, 'dataset': 'semi3d'},
      'fs':{'default_resolution':[128, 128], 'num_classes':1, 'dataset': 'fs'}

    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt