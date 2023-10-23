from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import os

import torch
import torch.utils.data
from cet_pick.opts import opts
from cet_pick.models.model import create_model, load_model, save_model, save_center, load_center
# from cmodels.data_parallel import DataParallel
from logger import Logger
from cet_pick.datasets.dataset_factory import get_dataset
from cet_pick.trains.train_factory import train_factory

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  print('model', model)
  # print(count_parameters(model))
  # first we freeze the last layers 
  for name, param in model.named_parameters():
    if 'hm' in name:
    # print(name)
      param.requires_grad = False
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)

  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  print('features without any training')
  epoch = 1
  cluster_centers = None 
  cluster_ind = None
  # with torch.no_grad():
  #   log_dict_val, preds = trainer.val(epoch, val_loader)
  # print('now start training ....')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _, = trainer.train(epoch, train_loader, pretrain=True)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      # save_center(os.path.join(opt.save_dir, 'center_{}.pth'.format(mark)), 
      #            epoch, cluster_centers[cluster_ind])
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch,val_loader, pretrain=True)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best_contrastive.pth'), 
                   epoch, model)
        # save_center(os.path.join(opt.save_dir, 'center_best_contrastive.pth'),
        #            epoch, cluster_centers[cluster_ind])
    else:
      save_model(os.path.join(opt.save_dir, 'model_last_contrastive.pth'), 
                 epoch, model, optimizer)
      # save_center(os.path.join(opt.save_dir, 'center_last_contrastive.pth'),
      #              epoch, cluster_centers[cluster_ind])
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  print('start training class layer ')
  for name, param in model.named_parameters():
    if not 'hm' in name:
    # print(name)
      param.requires_grad = False
    if 'hm' in name:
      param.requires_grad = True
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.device)
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _, = trainer.train(epoch,train_loader, pretrain=False)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      # save_center(os.path.join(opt.save_dir, 'center_{}.pth'.format(mark)), 
      #            epoch, cluster_centers[cluster_ind])
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader, pretrain=False)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best_contrastive.pth'), 
                   epoch, model)
        # save_center(os.path.join(opt.save_dir, 'center_best_contrastive.pth'),
        #            epoch, cluster_centers[cluster_ind])
    else:
      save_model(os.path.join(opt.save_dir, 'model_last_contrastive.pth'), 
                 epoch, model, optimizer)
      # save_center(os.path.join(opt.save_dir, 'center_last_contrastive.pth'),
      #              epoch, cluster_centers[cluster_ind])
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  # logger.close()


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)