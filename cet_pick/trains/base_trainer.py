from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from cet_pick.utils.utils import AverageMeter
from cet_pick.models.decode import tomo_decode
from cet_pick.models.utils import _center_distance
from cet_pick.models.loss import SupConLoss
from cet_pick.models.data_parallel import DataParallel

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model 
        self.loss = loss 

    def forward(self, batch, epoch, phase):
        # print('batch input', batch['input'].shape)
        outputs = self.model(batch['input'])
        # this part is only for contrastive use comment out if we are not doing contrastive learning
        if phase == 'train':
            outputs_cr = self.model(batch['input_aug'])
        else:
            outputs_cr = None
        loss, loss_stats = self.loss(outputs, batch, epoch, phase, output_cr=outputs_cr)
        return outputs[-1], loss, loss_stats 

class ModelWithLossCl(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLossCl, self).__init__()
        self.model = model 
        self.loss = loss 

    def forward(self, batch, epoch, phase, cluster_centers = None, cluster_ind = None):

        outputs = self.model(batch['input'])
        # this part is only for contrastive use comment out if we are not doing contrastive learning
        if phase == 'train':
            outputs_cr = self.model(batch['input_aug'])
        # else:
        #     outputs_cr = None
        loss, loss_stats, cluster_centers, cluster_ind = self.loss(outputs, batch, epoch, phase, outputs_cr=outputs_cr, cluster_center = cluster_centers, cluster_ind = cluster_ind)
        return outputs[-1], loss, loss_stats, cluster_centers, cluster_ind




class BaseTrainer(object):
    def __init__(
        self, opt, model, optimizer=None):
        self.opt= opt 
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)
        # self.model_with_loss = ModelWithLossCl(model, self.loss)

    def set_distributed_device(self, gpus):
        if gpus is not None:
            torch.cuda.set_device(gpus)
            self.model_with_loss.cuda(gpus)
            self.model_with_loss = torch.nn.parallel.DistributedDataParallel(self.model_with_loss, device_ids = [gpus])
            model_without_ddp = self.model_with_loss.module 
        else:
            self.model_with_loss.cuda()
            self.model_with_loss = torch.nn.parallel.DistributedDataParallel(self.model_with_loss)
            model_without_ddp = self.model_with_loss.module

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids = gpus, chunk_sizes = chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch_contrastive(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt 
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters 
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            # print('batch', batch['class'])
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            
            for k in batch:
                if k!= 'meta':
                    batch[k] = batch[k].to(device = opt.device, non_blocking=True)
        return

    def run_epoch_fewshot(self, phase, epoch, data_loader, cluster_centers = None, cluster_ind = None):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()
        # sup_loss = SupConLoss(0.2, 0.7, 0.7)
        opt = self.opt 
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters 
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            
            for k in batch:
                # print('k', k)
                if k!= 'meta' and k!='ml_graph' and k!='cl_graph':
                    batch[k] = batch[k].to(device = opt.device, non_blocking=True)
            output, loss, loss_stats, cluster_centers, cluster_ind = model_with_loss(batch, epoch, phase, cluster_centers, cluster_ind)
            loss = loss.mean()
            # torch.autograd.set_detect_anomaly(True)
            if phase == 'train':
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
            else:
                bar.next()
            if phase != 'train':
                if opt.debug > 0:
                    self.debug(batch, output, cluster_centers[cluster_ind], iter_id)

            if opt.test:
                self.save_result(output, batch, cluster_centers[cluster_ind], results)

            del output, loss, loss_stats
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        # ret['cluster_centers'] = cluster_centers
        # ret['cluster_ind'] = cluster_ind

        return ret, results, cluster_centers, cluster_ind

    def run_epoch(self, phase, epoch, data_loader):
        # print('epoch', epoch)
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()
        # sup_loss = SupConLoss(0.2, 0.7, 0.7)
        opt = self.opt 
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters 
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):

            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            
            for k in batch:
                if k!= 'meta':
                    if opt.distributed:
                    # batch[k] = batch[k].to(device = opt.device, non_blocking=True)
                        batch[k] = batch[k].cuda(opt.gpu)
                    else:
                        batch[k] = batch[k].to(device = opt.device, non_blocking=True)
                    # print(k, batch[k].shape)
                    # batch[k] = batch[k].to(device = opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch, epoch, phase)
            
            loss = loss.mean()
            # torch.autograd.set_detect_anomaly(True)
            if phase == 'train':
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
            else:
                bar.next()
            if phase != 'train':
                if opt.debug > 0:
                    self.debug(batch, output, iter_id)

            if opt.test:
                self.save_result(output, batch, results)

            del output, loss, loss_stats
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.


        return ret, results 

    def debug(self, batch, output, clustercenter, iter_id):
        raise NotImplementedError 

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError 

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)
        # return self.run_epoch_fewshot('val', epoch, data_loader, cluster_centers, cluster_ind)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
        # return self.run_epoch_fewshot('train', epoch, data_loader, cluster_centers, cluster_ind)


