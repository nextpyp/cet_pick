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
from cet_pick.utils.image import clip_img
import cv2
from cet_pick.models.decode import tomo_decode

from pytorch_metric_learning import miners, losses

class TomoDenoiseLoss(torch.nn.Module):
    """
    Trainer for PU Learner with Contrastive Regularization
    
    """
    def __init__(self, opt):
        super(TomoDenoiseLoss, self).__init__()
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
        # self.criterion= nn.CosineSimilarity(dim=1)
        
        self.opt = opt 

    def forward(self, noisy_in, mu, sigma, noise_std):

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
        # print('output_std', output_std)
        loss_out = ((noisy_in - mu) ** 2) / sigma + torch.log(sigma)
        final_loss = loss_out - 0.1 * noise_std
            # loss = hm_loss + cr_loss * self.opt.cr_weight + 0.1 * consis_loss
            # loss = hm_loss + cr_loss * self.opt.cr_weight
        
        final_loss = final_loss.view(final_loss.shape[0], -1).mean(1, keepdim=True)

        # loss_stats = {'loss': cos_loss,'hm_loss': hm_loss, 'cr_loss': cr_loss, 'consis_loss': consis_loss}
        # loss_stats = {'loss': loss,'hm_loss': hm_loss, 'consis_loss': consis_loss}
        loss_stats = {'loss': final_loss}
        return final_loss, loss_stats

class TomoDenoiseTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(TomoDenoiseTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss']
        # loss_states = ['loss','hm_loss', 'consis_loss']
        loss = TomoDenoiseLoss(opt)
        return loss_states, loss 

    def debug(self, batch, output, iter_id):
        opt = self.opt  
        mu_img = output['img_mu'].detach().cpu()
        std_img = output['model_std'].detach().cpu()
        dn_img = output['img_denoise'].detach().cpu()
        name = batch['meta']['name']
        tilt_num = batch['meta']['ind']
        noisy_ins = batch['noisy_in'].detach().cpu()
        gt_ins = batch['gt'].detach().cpu()
        num_of_ims = mu_img.shape[0]
        debugger = Debugger(dataset = opt.dataset, down_ratio = opt.down_ratio)
        for i in range(num_of_ims):
            curr_im = noisy_ins[i][0]
            mu_im = mu_img[i][0]
            dn_im = dn_img[i][0]
            curr_name = name[i]
            gt_im = gt_ins[i][0]
            print('curr_name', curr_name)
            tlt_num = tilt_num[i]
            print('tilt_num', tlt_num)
            mu_im = clip_img(mu_im)
            dn_im = clip_img(dn_im)
            curr_im = clip_img(curr_im)
            gt_im = clip_img(gt_im)
            mu_im = mu_im.numpy()
            dn_im = dn_im.numpy()
            curr_im = curr_im.numpy()
            gt_im = gt_im.numpy()
            print('curr_im', curr_im.shape)
            print('unpad im', gt_im.shape)
            mu_im = np.clip(mu_im * 255., 0, 255).astype(np.uint8)
            dn_im = np.clip(dn_im * 255., 0, 255).astype(np.uint8)
            curr_im = np.clip(curr_im * 255., 0, 255).astype(np.uint8)
            gt_im = np.clip(gt_im * 255., 0, 255).astype(np.uint8)
            debugger.add_slice(dn_im, 'dn_im')
            debugger.add_slice(mu_im, 'mu_im')
            debugger.add_slice(curr_im, 'noisy_im')
            debugger.add_slice(gt_im, 'unpad_im')
            prefix = '{}_{}'.format(name, iter_id)
            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix=prefix, slice_num = tlt_num)

    def save_results(self, output, batch, results):
        pass