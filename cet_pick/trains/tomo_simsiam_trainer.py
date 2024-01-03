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

class TomoSimSiamLoss(torch.nn.Module):
    """
    Trainer for PU Learner with Contrastive Regularization
    
    """
    def __init__(self, opt):
        super(TomoSimSiamLoss, self).__init__()
        self.criterion= nn.CosineSimilarity(dim=1)
        
        self.opt = opt 

    def forward(self, outputs, batch, epoch):

        opt = self.opt 

        p1, z1 = outputs[0]['pred'], outputs[0]['proj']
        p2, z2 = outputs[1]['pred'], outputs[1]['proj']
        cos_loss = -(self.criterion(p1, z2).mean()+self.criterion(p2, z1).mean())*0.5
        output = p1.detach()
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        loss_stats = {'loss': cos_loss, 'cosine_loss': cos_loss, 'output_std': output_std}
        return cos_loss, loss_stats

class TomoSimSiamTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(TomoSimSiamTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss','cosine_loss', 'output_std']
        loss = TomoSimSiamLoss(opt)
        return loss_states, loss 

    def debug(self, batch, output, iter_id):
        pass

    def save_results(self, output, batch, results):
        pass