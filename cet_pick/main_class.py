from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import os

import torch
import torch.utils.data
import numpy as np 
from cet_pick.opts import opts
from PIL import Image
from cet_pick.models.model import create_model, load_model, save_model, save_center, load_center
# from cmodels.data_parallel import DataParallel
from logger import Logger
import time
from cet_pick.datasets.dataset_factory import get_dataset
from cet_pick.trains.train_factory import train_factory
from cet_pick.utils.sampler import StratifiedCoordinateHMSampler

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

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  width_z, width_xy = model.width_z, model.width_xy

  if opt.distributed:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)


  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  if opt.distributed:
    trainer.set_distributed_device(opt.gpu)
  else:
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  # trainer.set_device(opt.gpus, opt.device)

  # print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val', width_xy, width_z), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  # if opt.test:
  #   _, preds = trainer.val(0, val_loader)
  #   val_loader.dataset.run_eval(preds, opt.save_dir)
  #   return
  dataset_train = Dataset(opt, 'train', width_xy, width_z)
  train_hms = dataset_train.hms 
  start_time = time.time()
  sampler = StratifiedCoordinateHMSampler(train_hms, size=1000*256, split='pu')
  end_time = time.time()

  train_loader = torch.utils.data.DataLoader(
      dataset_train, 
      batch_size=256, 
      num_workers=opt.num_workers,
      sampler=sampler
  )





  print('Starting training...')
  best = 1e10
  start_epoch = 0

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _, = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)

      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)

      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best_contrastive.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last_contrastive.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)