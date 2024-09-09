from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

# from cet_pick.models.loss import FocalLoss, RegL1Loss, RegLoss, BCELoss
from cet_pick.models.loss import FocalLoss, FocalLoss_mod, RegL1Loss, RegLoss, UnbiasedConLoss, ConsistencyLoss, PULoss, BiasedConLoss, SupConLossV2_more, PUGELoss, BCELoss
from cet_pick.models.decode import tomo_decode
from cet_pick.models.decode import _nms
from cet_pick.models.utils import _sigmoid 
from cet_pick.trains.base_trainer import BaseTrainer 
from cet_pick.utils.debugger import Debugger
from cet_pick.utils.post_process import tomo_post_process
import cv2

class TomoCRClassLoss(torch.nn.Module):
    def __init__(self, opt):
        super(TomoCRClassLoss, self).__init__()
        self.bce = BCELoss()


        self.opt = opt 
        if opt.pn:
            self.crit = FocalLoss()

        if opt.ge:
            criteria = FocalLoss_mod(opt.thresh)

            # self.crit = PUGELoss(opt.tau, criteria=self.bce)
            self.crit = PUGELoss(opt.tau, criteria=criteria)
        if opt.pn:
            self.cr_loss = SupConLossV2_more(opt.temp)
        else:
            self.cr_loss = UnbiasedConLoss(opt.temp, opt.tau)

        self.crit2 = FocalLoss()
        
        self.cons_loss = ConsistencyLoss()



    def forward(self, outputs, batch, epoch, phase, output_cr = None):
        opt = self.opt 

        cr_loss, hm_loss, consis_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if output_cr is not None:
                output_cr = output_cr[s]
            # if phase != "train":
            if 1:

                output['hm'] = _sigmoid(output['hm'])

        if phase == 'train':
            hm_loss += self.crit(output['hm'], batch['label']) / opt.num_stacks 
        else:
            hm_loss += self.crit2(output['hm'], batch['hm']) / opt.num_stacks

        if opt.contrastive and phase == "train":
            output_fm = output['proj']
            b, ch, d, w, h = output_fm.shape
            gt_hm = batch['label']
            output_fm_cr = output_cr['proj']
            output_hm_cr = output_cr['hm']
            output_fm = output_fm.reshape(b, ch, -1).contiguous()
            output_fm = output_fm.permute(1,0,2)
            output_fm = output_fm.reshape(ch, -1).T
            output_fm_cr = output_fm_cr.reshape(b, ch, -1).contiguous()
            output_fm_cr = output_fm_cr.permute(1,0,2)
            output_fm_cr = output_fm_cr.reshape(ch, -1).T
            gt_hm_f = gt_hm.reshape(b, -1).contiguous()
            # gt_hm_f = gt_hm.reshape(1, -1)
            gt_hm_f = gt_hm_f.reshape(-1).contiguous()
            output_hm = output['hm'].reshape(b, -1).contiguous()
            # output_hm = output_hm.squeeze()
            output_hm = output_hm.reshape(-1).contiguous()
            output_hm_cr = output_hm_cr.reshape(b, -1).contiguous()
            # output_cr = output_cr.squeeze()
            output_hm_cr = output_hm_cr.reshape(-1).contiguous()

            if self.opt.pn:
                loss_cr = self.cr_loss(gt_hm_f, output_hm, output_hm_cr, output_fm, output_fm_cr, opt)
                cr_loss += loss_cr
            else:
                debiased_loss_sup, debiased_loss_unsup = self.cr_loss(gt_hm_f, output_hm, output_hm_cr, output_fm, output_fm_cr, opt)

                cr_loss += debiased_loss_sup + 0.1*debiased_loss_unsup
            
            consis_loss += self.cons_loss(output_hm, output_hm_cr)
            loss = hm_loss + cr_loss * self.opt.cr_weight + consis_loss
            # loss = hm_loss

        else:
            cr_loss = hm_loss * 0
            loss = hm_loss
            consis_loss = hm_loss * 0


        loss_stats = {'loss': loss,'hm_loss': hm_loss, 'cr_loss': cr_loss, 'consis_loss': consis_loss}
        return loss, loss_stats

            # if not opt.mse_loss:
                # output['hm'] = _sigmoid(output['hm'])
                # output['class'] = 
        # bce_loss += self.bce(output['class'], batch['class'])

        loss_stats = {'loss': bce_loss}

        return bce_loss, loss_stats

class TomoCRClassTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(TomoCRClassTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss','hm_loss','cr_loss', 'consis_loss']
        loss = TomoCRClassLoss(opt)
        return loss_states, loss 

    def debug(self, batch, output, iter_id):
        opt = self.opt 
        dets = tomo_decode(output['hm'], reg=None, K = opt.K, if_fiber = opt.fiber)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        post_dets = tomo_post_process(dets, z_dim_tot = 128)
        dets_gt = batch['gt_det'].detach().cpu().numpy().reshape(1, -1, 3)
        name = batch['meta']['name']
        post_dets_gt = tomo_post_process(dets_gt)
        for i in range(1):
            debugger = Debugger(dataset = opt.dataset, down_ratio = opt.down_ratio)
            tomo = batch['input'][i].detach().cpu().numpy()
            hm_ = output['hm'][i].detach().cpu().numpy()[0]
            # print('hm_shape', hm_.shape)
            hm_zmax = hm_.shape[0]
            gts = post_dets_gt[i]
            preds = post_dets[i]
            name = name[i]
            det_zs = preds.keys()
            debugger.save_detection(preds, path = opt.debug_dir, prefix = iter_id, name=name)
            for z in np.arange(20, hm_zmax-20):
                out_z = output['hm'][i].detach().cpu().numpy()[0][z]
                out_z_n = output['hm'][i]
                out_z_nms = _nms(out_z_n)
                out_z_nms = out_z_nms.detach().cpu().numpy()[0][z]
                out_z_gt = batch['hm'][i].detach().cpu().numpy()[z]
                out_z = np.expand_dims(out_z, 0)
                out_z_gt = np.expand_dims(out_z_gt, 0)
                pred = debugger.gen_colormap(out_z)
                gt = debugger.gen_colormap(out_z_gt)
                tomo_z = cv2.normalize(tomo[z], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # out_p = cv2.normalize(out_proj, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                tomo_z = np.clip(tomo_z * 255., 0, 255).astype(np.uint8)
                # print('tomo_z', tomo_z.shape)
                img_slice = np.dstack((tomo_z, tomo_z, tomo_z))
                debugger.add_slice(pred, 'pred_hm')
                # print('pred hm', pred.shape)
                debugger.add_blend_img(img_slice, gt, 'gt_hm')
                debugger.add_slice(img_slice, img_id = 'pred_out')
                debugger.add_slice(img_slice, img_id = 'gt_out')
                
                if z in preds.keys():
                    slice_coords = preds[z]
                    debugger.add_particle_detection(slice_coords, 8, img_id = 'pred_out')
                if z in gts.keys():
                    slice_coords = gts[z]
                    debugger.add_particle_detection(slice_coords, 8, img_id = 'gt_out')
                if opt.debug == 4:
                    debugger.save_all_imgs(opt.debug_dir, prefix='{}_{}'.format(iter_id, name), slice_num = z)

        
  


    def save_results(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None 
        dets = tomo_decode(output['hm'], output['wh'], reg=reg, K = self.opt.K, if_fiber = self.opt.fiber)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        return dets 