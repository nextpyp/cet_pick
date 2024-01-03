from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import os

import torch
import torch.utils.data
from cet_pick.opts import opts
import numpy as np 
import random
from cet_pick.models.model import create_model, load_model, save_model, save_center, load_center
# from cmodels.data_parallel import DataParallel
from cet_pick.trains.tomo_moco_small_trainer import MoCoModel, MoCoTrainer
from logger import Logger
from cet_pick.datasets.dataset_factory import get_dataset
from cet_pick.trains.train_factory import train_factory
import torch.distributed as dist
from cet_pick.utils.utils import adjust_learning_rate, warmup_learning_rate

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(opt):
  torch.manual_seed(opt.seed)
  # torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  if "WORLD_SIZE" in os.environ:
    opt.world_size = int(os.environ["WORLD_SIZE"])
  print('opt.world', opt.world_size)
  opt.distributed = opt.world_size > 1
  print('distributed', opt.distributed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  ngpus_per_node = torch.cuda.device_count()
  if opt.distributed:
    if opt.local_rank != -1: # for torch.distributed.launch
      opt.rank = opt.local_rank
      opt.gpu = opt.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
      opt.rank = int(os.environ['SLURM_PROCID'])
      opt.gpu = opt.rank % torch.cuda.device_count()
    dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                            world_size=opt.world_size, rank=opt.rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  if opt.distributed:
    if opt.rank == 0:
      logger = Logger(opt)
  else:
    logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model_q = create_model(opt.arch, opt.heads, opt.head_conv)
  model_k = create_model(opt.arch, opt.heads, opt.head_conv)
  if opt.distributed:
    model_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_q)
    model_k = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_k)
  # print('model', model_q)
  model = MoCoModel(model_q, model_k,opt)
  print('model', model.encoder_q)
  # print(count_parameters(model))
  # first we freeze the last layers 
  # for name, param in model.parameters():
  #   if 'out' in name:
      # param.requires_grad = False
  # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)
  init_lr = opt.lr * opt.batch_size/256
  optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)

  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  # Trainer = train_factory[opt.task]
  trainer = MoCoTrainer(opt, model, optimizer)
  if opt.distributed:
    trainer.set_distributed_device(opt.gpu)
  else:
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  # trainer.set_device(opt.gpus, opt.device)

  print('Setting up data...')
  # val_loader = torch.utils.data.DataLoader(
  #     Dataset(opt, 'val'), 
  #     batch_size=1, 
  #     shuffle=False,
  #     num_workers=1,
  #     pin_memory=True
  # )

  # if opt.test:
  #   _, preds = trainer.val(0, val_loader)
  #   val_loader.dataset.run_eval(preds, opt.save_dir)
  #   return

  # train_loader = torch.utils.data.DataLoader(
  #     Dataset(opt, 'train'), 
  #     batch_size=opt.batch_size, 
  #     shuffle=True,
  #     num_workers=opt.num_workers,
  #     pin_memory=True,
  #     drop_last=True
  # )
  dataset = Dataset(opt, 'train', (3, opt.bbox, opt.bbox), sigma1=[2.5,5], K = 16000)
  if opt.distributed:
    # train_sampler = torch.utils.data.distributed.DistributedSampler(Dataset(opt, 'train',(3,48,48)), shuffle=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler = train_sampler,
        drop_last=True
    )
  else:
    train_sampler=None
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=(train_sampler is None),
        # shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

  dat = next(iter(train_loader))
  aug1 = dat['input']
  # aug13d = dat['input_3d']
  aug2 = dat['input_aug']
  # aug3_3d = dat['input_aug_3d']
  # coord = dat['coord']
  np.save('aug1.npy', aug1.numpy())
  np.save('aug2.npy', aug2.numpy())
  # np.save('aug1_3d.npy', aug13d.numpy())
  # np.save('aug2_3d.npy', aug3_3d.numpy())
  # np.save('coords.npy', coord.numpy())
  # print('dat', dat)
  print('Starting training...')
  # print('batch_size', opt.batch_size)
  best = 1e10
  epoch = 1
  # cluster_centers = None 
  # cluster_ind = None
  # # with torch.no_grad():
  # #   log_dict_val, preds = trainer.val(epoch, val_loader)
  # # print('now start training ....')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    np.random.seed(epoch)
    random.seed(epoch)
    if opt.distributed: 
      train_loader.sampler.set_epoch(epoch)
    adjust_learning_rate(opt, optimizer, epoch)
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _, = trainer.train(epoch, train_loader)
    if opt.distributed:
      if opt.rank == 0:
        logger.write('epoch: {} |'.format(epoch))
    else:
      logger.write('epoch: {} |'.format(epoch))
    if opt.distributed:
      if opt.rank == 0:
        for k, v in log_dict_train.items():
          logger.scalar_summary('train_{}'.format(k), v, epoch)
          logger.write('{} {:8f} | '.format(k, v))
    else:
      for k, v in log_dict_train.items():
          logger.scalar_summary('train_{}'.format(k), v, epoch)
          logger.write('{} {:8f} | '.format(k, v))
    if opt.distributed:
      if opt.val_intervals > 0 and epoch % opt.val_intervals == 0 and opt.rank == 0:
      # if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                   epoch, model, optimizer)
      
    elif not opt.distributed:
      if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                   epoch, model, optimizer)

      # with torch.no_grad():
      #   log_dict_val, preds = trainer.val(epoch, val_loader)
      # for k, v in log_dict_val.items():
      #   logger.scalar_summary('val_{}'.format(k), v, epoch)
      #   logger.write('{} {:8f} | '.format(k, v))
      # if log_dict_val[opt.metric] < best:
      #   best = log_dict_val[opt.metric]
      #   save_model(os.path.join(opt.save_dir, 'model_best_contrastive.pth'), 
      #              epoch, model)
      else:
        save_model(os.path.join(opt.save_dir, 'model_last_moco.pth'), 
                   epoch, model, optimizer)
    if opt.distributed:
      if opt.rank == 0:
        logger.write('\n')
    else:
      logger.write('\n')
    if epoch in opt.lr_step:
      if opt.distributed:
        if opt.rank == 0:
          save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                     epoch, model, optimizer)
      else:

        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                   epoch, model, optimizer)
      # lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      # print('Drop LR to', lr)
      # for param_group in optimizer.param_groups:
          # param_group['lr'] = lr
  if opt.distributed:
    if opt.rank == 0:  
      logger.close()
  else:
    logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)