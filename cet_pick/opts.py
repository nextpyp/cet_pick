from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

  
def list_of_floats(arg):
  return list(map(float, arg.split(',')))
  
class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('task', default='semi',
                             help='particle detection for cryoET/tomo datasets, ' 
                                  'for semi-supervised/few shot particle detection, please use semi, '
                                  'for unsupervised pretraining, please use simsiam, '
                                  'for unsupervised evaluation, please use simsiam_test. ')
    self.parser.add_argument('--dataset', default='semi',
                             help='cryoET dataset has default tomo'
                                  'for semi-supervised/few shot particle detection during training, please use semi, '
                                  'for semi-supervised/few shot particle detection during evaluation, please use semi_test, ')
    self.parser.add_argument('--exp_id', default='default', help='experiment id for this run to save all outputs')
    self.parser.add_argument('--test', action='store_true', help='whether to perform testing after training')
    self.parser.add_argument('--debug', type=int, default=4,
                             help=' for semi-supervised detection, 4: save all visualizations to disk, including heatmaps, ground truth annotations')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--pretrain_model', default='', help='path to simsiam pretrain weights')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 
    self.parser.add_argument('--fiber', action='store_true', help='whether trying to identify fiber. changes in postprocessing and initialization')

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
                             help='save model to disk every x epochs.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    
    # model
    self.parser.add_argument('--arch', default='unet_4', 
                             help='model architecture. Currently tested'
                                  'ressmall_18 | unet_4 | unet_5 ,'
                                  'unet_x typically performs better than ressmall')
    self.parser.add_argument('--last_k', type=int, default=3,
                             help='kernel size for the last convolution prior to projection layer, default is 3,' 
                             'change this to bigger number if particle is bigger')
    
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '32 for particle identification and 128 for simsiam training.')
    self.parser.add_argument('--down_ratio', type=int, default=2,
                             help='output stride. Currently only supports 2 for particle detection.')

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

    self.parser.add_argument('--contrastive', action='store_true',
                             help='whether contrastive training')

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
    self.parser.add_argument('--nclusters', type=int, default=3, help='number of clusters for SCAN')
    self.parser.add_argument('--nheads', type=int, default=1, help='number of heads for SCAN model')
    self.parser.add_argument('--names', type=str, help='list of names of tomograms')
    # self.parser.add_argument('--name', type=str, hlep='single na')
    # test

    self.parser.add_argument('--nms', type=int, default=3, help='radius for running nms, default is 3')
    self.parser.add_argument('--cutoff_z', type=int, default=10, help='removing # of leading and trailing z slices')
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

    

    self.parser.add_argument('--pn', action='store_true',
                             help='whether to use positive negative learning with fully labeled data')
    self.parser.add_argument('--ge', action='store_true',
                             help='whether to use generalized criteria loss')
    ### fiber specific 
    self.parser.add_argument('--distance_cutoff', type=float, default=15, help='distance cutoff for whether two points are connected in graph')
    self.parser.add_argument('--r2_cutoff', type=float, default=30, help='max residual for fitted curve, omit if above the residual/bad fitting')
    self.parser.add_argument('--curvature_cutoff', type=float, default=0.003, help='max curvature for fitted curve, for microtubules the curvature should be small')
    ### data 

    self.parser.add_argument('--train_img_txt', type=str, default='train_images.txt', help='path to file of training images')
    self.parser.add_argument('--train_coord_txt', type=str, default='train_coords.txt', help='path to file of training coords')
    self.parser.add_argument('--val_img_txt', type=str, default='val_images.txt', help='path to file of validation images')
    self.parser.add_argument('--val_coord_txt', type=str, default='val_coords.txt', help='path to file of validation coordinates')
    self.parser.add_argument('--test_img_txt', type=str, default = 'test_images.txt', help='path to file of test images')
    self.parser.add_argument('--test_coord_txt', type=str, default='test_coords.txt', help='path to file of test coords')
    self.parser.add_argument('--compress', action='store_true', 
                              help = 'whether to combine 2 slice into 1 slice during reading of the dataset')
    self.parser.add_argument('--gauss', type=float, default=0,
                              help = 'whether to gaussian filter input tomogram during reading of the dataset and the value for sigma')
    self.parser.add_argument('--cluster_head', action='store_true',
                              help = 'whether to update cluster head only for SCAN')
    self.parser.add_argument('--out_id', type=str, default='output', help = 'directory name for all evaluation outputs. ')
    self.parser.add_argument('--order', type=str, default='xzy', help='input order for reconstructed tomogram')
    self.parser.add_argument('--dog', type=list_of_floats, default=[2.5,5], help='gaussian sigma for difference of gaussian operation')



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
      if opt.task == 'simsiam' or opt.task == 'simsiam2d3d' or opt.task == 'simsiam3d':
        opt.head_conv = 128 
      if opt.task =='semi':
        opt.head_conv = 32
      # opt.head_conv = 256 if 'dla' in opt.arch else 16
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
      # opt.batch_size = 1
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
    print('Training chunk_sizes:', opt.chunk_sizes)
    # need opt root dir
    # opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.root_dir = os.getcwd()
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    if opt.task == 'scan2d3d':
      opt.simsiam_dir = os.path.join(opt.root_dir, 'exp', 'simsiam2d3d', opt.exp_id)
    elif opt.task == 'scan':
      opt.simsiam_dir = os.path.join(opt.root_dir, 'exp', 'simsiam', opt.exp_id)
    opt.out_path = os.path.join(opt.save_dir, opt.out_id)
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
      opt.heads = {'hm': opt.num_classes, 'proj':opt.head_conv}
    elif opt.task == 'fs':
      opt.heads = {'proj':16}
    elif opt.task == 'tcla':
      opt.heads ={'class':1}
    elif opt.task == 'simsiam' or opt.task == 'simsiam2d3d' or opt.task == 'simsiam3d':
      opt.heads = {'proj': opt.head_conv, 'pred': opt.head_conv}
    elif opt.task == 'scan' or opt.task == 'scan2d3d':
      opt.heads = {'proj': opt.head_conv, 'pred': opt.head_conv}

    elif opt.task == 'moco':
      opt.heads = {'proj': 256, 'pred': 256}
    elif opt.task == 'denoise':
      opt.heads = {'proj': 128}
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    default_dataset_info = {
     
      'tomo': {'default_resolution': [512, 512], 'num_classes': 1, 'dataset': 'tomo'},
      'cr':{'default_resolution':[64, 64], 'num_classes':1, 'dataset': 'cr'},
      'semi': {'default_resolution':[64, 64], 'num_classes':1, 'dataset': 'semi'},
      'semi3d': {'default_resolution':[64, 64], 'num_classes':1, 'dataset': 'semi3d'},
      'fs':{'default_resolution':[128, 128], 'num_classes':1, 'dataset': 'fs'},
      'simsiam':{'default_resolution':[24, 24], 'num_classes':256, 'dataset':'simsiam'},
      'scan':{'default_resolution':[24, 24], 'num_classes':256, 'dataset':'scan'},
      'denoise':{'default_resolution':[64, 64], 'num_classes':256, 'dataset':'denoise'},
      'moco':{'default_resolution':[32, 32], 'num_classes':256, 'dataset':'moco'},

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