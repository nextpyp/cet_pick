from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn 
from cet_pick.models.loss import FocalLoss, RegL1Loss, RegLoss, UnbiasedConLoss, ConsistencyLoss, PULoss, BiasedConLoss, SupConLossV2_more, PUGELoss
from cet_pick.models.decode import _nms
from cet_pick.models.utils import _sigmoid 
from cet_pick.trains.base_trainer import BaseTrainer 
from cet_pick.utils.debugger import Debugger
from cet_pick.utils.post_process import tomo_post_process
import cv2
from cet_pick.models.decode import tomo_decode

from pytorch_metric_learning import miners, losses

class TomoMocoLoss(torch.nn.Module):
    """
    Trainer for PU Learner with Contrastive Regularization
    
    """
    def __init__(self, opt):
        super(TomoMocoLoss, self).__init__()
        # self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        # self.crit = torch.nn.MSELoss() if opt.mse_loss else PULoss(opt.tau)
        # self.crit2 = FocalLoss()
        # self.crit = PULoss(opt.tau)
        # # self.crit2 = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()

        # self.cr_loss = UnbiasedConLoss(opt.temp, opt.tau)
        # if opt.pn:
        #     # print('using pn loss')
        #     self.crit = FocalLoss()
        # elif opt.ge:
        #     criteria = FocalLoss()
        #     self.crit = PUGELoss(opt.tau, criteria=criteria)
        # else:
        #     self.crit = PULoss(opt.tau)
        # self.crit2 = FocalLoss()
        
        # self.crit2 = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        # if opt.pn:
        #     # print('using pn cr loss')
        #     self.cr_loss = SupConLossV2_more(opt.temp)
        # else:
        #     self.cr_loss = UnbiasedConLoss(opt.temp, opt.tau)
        # self.cr_loss = SupConLossV2_more(0.07,0.07,0.07)
        # self.cr_loss = losses.TripletMarginLoss()
        # self.cr_loss = BiasedConLoss(opt.temp)
        # self.unsup_cr_loss = UnSupConLoss(0.07)
        # self.cons_loss = ConsistencyLoss()
        self.criterion= nn.CrossEntropyLoss()
        
        self.opt = opt 

    def forward(self, outputs, batch, epoch):

        opt = self.opt 

        # cr_loss, hm_loss, consis_loss = 0, 0, 0
        # cos_loss = 0 
        # p1, z1 = outputs[0]['pred'], outputs[0]['proj']
        # p2, z2 = outputs[1]['pred'], outputs[1]['proj']
        # cos_loss = -(self.criterion(p1, z2).mean()+self.criterion(p2, z1).mean())*0.5
        # print('cos_loss', cos_loss.item())
        # output = p1.detach()
        # output = torch.nn.functional.normalize(output, dim=1)
        # output_std = torch.std(output, 0)
        # output_std = output_std.mean()
            # loss = hm_loss + cr_loss * self.opt.cr_weight + 0.1 * consis_loss
            # loss = hm_loss + cr_loss * self.opt.cr_weight
       
        loss = self.criterion(outputs[0], outputs[1])
        # loss_stats = {'loss': cos_loss,'hm_loss': hm_loss, 'cr_loss': cr_loss, 'consis_loss': consis_loss}
        # loss_stats = {'loss': loss,'hm_loss': hm_loss, 'consis_loss': consis_loss}
        loss_stats = {'loss': loss, 'infoNCE': loss}
        return loss, loss_stats

class TomoMocoTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(TomoMocoTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss','infoNCE']
        # loss_states = ['loss','hm_loss', 'consis_loss']
        loss = TomoMocoLoss(opt)
        return loss_states, loss 

    def debug(self, batch, output, iter_id):
        pass

    def save_results(self, output, batch, results):
        pass