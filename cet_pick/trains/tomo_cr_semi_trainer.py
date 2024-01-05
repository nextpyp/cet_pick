from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from cet_pick.models.loss import FocalLoss, RegL1Loss, RegLoss, UnbiasedConLoss, ConsistencyLoss, PULoss, BiasedConLoss, SupConLossV2_more, PUGELoss
from cet_pick.models.decode import _nms
from cet_pick.models.utils import _sigmoid 
from cet_pick.trains.base_trainer import BaseTrainer 
from cet_pick.utils.debugger import Debugger
from cet_pick.utils.post_process import tomo_post_process
import cv2
from cet_pick.models.decode import tomo_decode

class TomoCRSemiLoss(torch.nn.Module):
    """
    Trainer for PU Learner with Contrastive Regularization
    
    """
    def __init__(self, opt):
        super(TomoCRSemiLoss, self).__init__()
    
        if opt.pn:
            self.crit = FocalLoss()
        elif opt.ge:
            criteria = FocalLoss()
            self.crit = PUGELoss(opt.tau, criteria=criteria)
        else:
            self.crit = PULoss(opt.tau)
        self.crit2 = FocalLoss()

        if opt.pn:
            self.cr_loss = SupConLossV2_more(opt.temp)
        else:
            self.cr_loss = UnbiasedConLoss(opt.temp, opt.tau)

        self.cons_loss = ConsistencyLoss()
        
        self.opt = opt 

    def forward(self, outputs, batch, epoch, phase, output_cr = None):

        opt = self.opt 

        cr_loss, hm_loss, consis_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if output_cr is not None:
                output_cr = output_cr[s]
            if 1:
                output['hm'] = _sigmoid(output['hm'])
                if output_cr is not None:
                    output_cr['hm'] = _sigmoid(output_cr['hm'])

        if phase == 'train':
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks 
        else:
            hm_loss += self.crit2(output['hm'], batch['hm']) / opt.num_stacks
        # hm_loss += self.crit(output_cr['hm'], batch['hm_c']) / opt.num_stacks
        if opt.contrastive and phase == "train":
            output_fm = output['proj']
            b, ch, d, w, h = output_fm.shape
            gt_hm = batch['hm']
            output_fm_cr = output_cr['proj']
            output_hm_cr = output_cr['hm']
            flip_prob = batch['flip_prob']
            rot = batch['rot']
            diff_rot = 4-rot 
            # output_fm_cr = torch.rot90(output_fm_cr, k=diff_rot.item(), dims=[3,4])
            # output_hm_cr = torch.rot90(output_hm_cr, k=diff_rot.item(), dims=[3,4])
            if flip_prob > 0.5:
                output_fm_cr = output_fm_cr.flip(-2)
                output_cr = output_hm_cr.flip(-2)
            else:
                output_fm_cr = output_fm_cr.flip(-1)
                output_cr= output_hm_cr.flip(-1)
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
            # print('reshaped', output_hm.shape)
            output_cr = output_cr.reshape(b, -1).contiguous()
            # output_cr = output_cr.squeeze()
            output_cr = output_cr.reshape(-1).contiguous()
            if self.opt.pn:
                loss_cr = self.cr_loss(gt_hm_f, output_hm, output_cr, output_fm, output_fm_cr, opt)
                cr_loss += loss_cr
            else:
                debiased_loss_sup, debiased_loss_unsup = self.cr_loss(gt_hm_f, output_hm, output_cr, output_fm, output_fm_cr, opt)
            # biased_loss_sup, biased_loss_unsup = self.cr_loss(gt_hm_f, output_fm, output_fm_cr, opt)
            
                cr_loss += debiased_loss_sup + 0.1*debiased_loss_unsup
            # debiased_loss_sup, debiased_loss_unsup = self.cr_loss(gt_hm_f, output_hm, output_cr, output_fm, output_fm_cr, opt)
            # biased_loss_sup, biased_loss_unsup = self.cr_loss(gt_hm_f, output_fm, output_fm_cr, opt)
            consis_loss += self.cons_loss(output_hm, output_cr)
            loss = hm_loss + cr_loss * self.opt.cr_weight + consis_loss
        else:
            cr_loss = hm_loss * 0
            loss = hm_loss
            consis_loss = hm_loss * 0


        loss_stats = {'loss': loss,'hm_loss': hm_loss, 'cr_loss': cr_loss, 'consis_loss': consis_loss}
        return loss, loss_stats

class TomoCRSemiTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(TomoCRSemiTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss','hm_loss','cr_loss', 'consis_loss']
        loss = TomoCRSemiLoss(opt)
        return loss_states, loss 

    def debug(self, batch, output, iter_id):
        opt = self.opt 
        dets = tomo_decode(output['hm'], reg=None, K = opt.K, if_fiber = opt.fiber)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        if opt.task == 'semi3d':
            dets[:,:,:] *= opt.down_ratio
        else:
            dets[:,:,:2] *= opt.down_ratio 
        post_dets = tomo_post_process(dets, z_dim_tot = 128)
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, 3)
        name = batch['meta']['name']
        if opt.task == 'semi3d':
            dets_gt[:,:,:] *= opt.down_ratio 
        else:

            dets_gt[:,:,:2] *= opt.down_ratio 
        post_dets_gt = tomo_post_process(dets_gt)
        for i in range(1):
            debugger = Debugger(dataset = opt.dataset, down_ratio = opt.down_ratio)
            tomo = batch['input'][i].detach().cpu().numpy()
            hm_ = output['hm'][i].detach().cpu().numpy()[0]
            hm_zmax = hm_.shape[0]
            gts = post_dets_gt[i]
            preds = post_dets[i]
            name = name[i]
            det_zs = preds.keys()
            debugger.save_detection(preds, path = opt.debug_dir, prefix = iter_id, name=name)
            for z in np.arange(20, hm_zmax-20):
                if opt.task == 'semi3d':
                    # z = int(z//2)
                    out_z = output['hm'][i].detach().cpu().numpy()[0][int(z//2)]
                    out_z_n = output['hm'][i]
                    out_z_nms = _nms(out_z_n)
                    out_z_nms = out_z_nms.detach().cpu().numpy()[0][int(z//2)]
                    out_z_gt = batch['hm'][i].detach().cpu().numpy()[int(z//2)]
                else:
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

                img_slice = np.dstack((tomo_z, tomo_z, tomo_z))
                debugger.add_slice(pred, 'pred_hm')

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
