from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time
import numpy as np
import torch.nn as nn 
from cet_pick.models.loss import FocalLoss, RegL1Loss, RegLoss, UnbiasedConLoss, ConsistencyLoss, PULoss, BiasedConLoss, SupConLossV2_more, PUGELoss
from cet_pick.models.decode import _nms
from progress.bar import Bar
from cet_pick.utils.utils import AverageMeter
from cet_pick.models.utils import _sigmoid 
from cet_pick.trains.base_trainer import BaseTrainer 
from cet_pick.utils.debugger import Debugger
from cet_pick.utils.post_process import tomo_post_process
import cv2
from cet_pick.models.decode import tomo_decode
from cet_pick.models.data_parallel import DataParallel
from cet_pick.utils.utils import adjust_learning_rate, warmup_learning_rate, compute_ramped_lrate, adjust_lr_denoise

from pytorch_metric_learning import miners, losses

class MoCoModel(torch.nn.Module):
    def __init__(self, model_q, model_k, opt, dim=256, K=4096, m=0.99, T=0.1, bn_splits=8, symmetric=True):
        super(MoCoModel, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        self.encoder_q = model_q 
        self.encoder_k = model_k 
        # for name, param in self.encoder_q.named_parameters():
        #     print('name', name)
        # dim_mlp = self.encoder_q.Linear.weight.shape[1]
        # self.encoder_q.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
        #     )
        # self.encoder_k.fc = nn.Sequential(
        #     nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
        # )
        self.opt = opt 
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False

         # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]
    # def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    #     mlp = []
    #     for l in range(num_layers):
    #         dim1 = input_dim if l == 0 else mlp_dim
    #         dim2 = output_dim if l == num_layers - 1 else mlp_dim

    #         mlp.append(nn.Linear(dim1, dim2, bias=False))

    #         if l < num_layers - 1:
    #             mlp.append(nn.BatchNorm1d(dim2))
    #             mlp.append(nn.ReLU(inplace=True))
    #         elif last_bn:
    #             # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
    #             # for simplicity, we further removed gamma in BN
    #             mlp.append(nn.BatchNorm1d(dim2, affine=False))

    #     return nn.Sequential(*mlp)

    def contrastive_loss(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, q, k


    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        # loss_stats = {}
        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        loss_stats = {'loss': loss, 'moco_loss': loss}

        return loss, loss_stats


class MoCoTrainer(object):
    def __init__(self, opt, model, optimizer):
        self.opt= opt 
        self.optimizer = optimizer

        # self.model_with_loss = MoCoModel(model_q, model_k, dim=256, K=4096, m=0.99, T=0.1, bn_splits=8, symmetric=True)
        self.model_with_loss = model
        self.loss_stats = ['loss', 'moco_loss']
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

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()
        opt = self.opt 
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters 
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        results = {}
        for iter_id, batch in enumerate(data_loader):
            # actual_iter = iter
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

            loss, loss_stats = model_with_loss(batch['input'], batch['input_aug'])
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

            del loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.


        return ret, results

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
