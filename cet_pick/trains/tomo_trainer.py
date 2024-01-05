from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from cet_pick.models.loss import FocalLoss, RegL1Loss, RegLoss, SupConLoss, UnSupConLoss
from cet_pick.models.decode import tomo_decode
from cet_pick.models.decode import _nms
from cet_pick.models.utils import _sigmoid 
from cet_pick.trains.base_trainer import BaseTrainer 
from cet_pick.utils.debugger import Debugger
from cet_pick.utils.post_process import tomo_post_process
import cv2

class TomoLoss(torch.nn.Module):
    def __init__(self, opt):
        super(TomoLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else RegLoss() if opt.reg_loss == 'sl1' else None 
        self.crit_wh = self.crit_reg 
        self.cr_loss = SupConLoss(0.2, 0.2, 0.2)
        # self.cr_loss = losses.TripletMarginLoss()
        self.unsup_cr_loss = UnSupConLoss(0.2)
        self.opt = opt 

    def forward(self, outputs, batch, epoch, phase):
        opt = self.opt 
        hm_loss, wh_loss, off_loss, cr_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])
                out_mask = _sigmoid(7*(output['hm']-0.5))


        hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks 
        if opt.contrastive and phase == "train":
            output_fm = output['proj']
            gt_dets = batch['gt_det']
            pos_features = []
            labels = []
            for j in range(gt_dets.shape[1]):
                curr_pos = gt_dets[0][j].long()
                # print('curr_pos', curr_pos)
                curr_feat = output_fm[0,:,curr_pos[-1], curr_pos[1], curr_pos[0]]
                # print('curr_feat', curr_feat.shape)
                pos_features.append(curr_feat.unsqueeze(0))
                # labels.append(0)
            # pos_features = torch.cat(pos_features, dim=0)
            # print('all_pos_features', pos_features.shape)
            # total_pos_features = pos_features.shape[0]
            hard_neg_features = []
            soft_pos = batch['soft_neg']
            for j in range(soft_pos.shape[1]):
                curr_pos = soft_pos[0][j].long()
                curr_feat = output_fm[0,:,curr_pos[-1], curr_pos[1], curr_pos[0]]
                # soft_neg_features.append(curr_feat.unsqueeze(0))
                pos_features.append(curr_feat.unsqueeze(0))
                # labels.append(1)
            pos_features = torch.cat(pos_features, dim=0)
            print('all_pos_features', pos_features.shape)
            total_pos_features = pos_features.shape[0]
            # soft_neg_features = torch.cat(soft_neg_features, dim=0)
            # print('soft_neg_features', soft_neg_features.shape)
            # total_neg_features = soft_neg_features.shape[0]
            hard_negs = batch['hard_neg']
            hard_neg_features = []
            for j in range(hard_negs.shape[1]):
                curr_pos = hard_negs[0][j].long()
                curr_feat = output_fm[0,:,curr_pos[-1], curr_pos[1], curr_pos[0]]
                hard_neg_features.append(curr_feat.unsqueeze(0))
            #     labels.append(2)
            hard_neg_features = torch.cat(hard_neg_features, dim=0)
            cr_loss += self.cr_loss(pos_features, hard_neg_features, opt)
            if epoch > 0:
                
            # all_labels = torch.tensor(labels).to(device = opt.device)
                all_features = torch.cat((pos_features, hard_neg_features), dim = 0)
                torch.cuda.empty_cache()
                unsup_cr_loss_pos = self.unsup_cr_loss(all_features, output_fm, out_mask,batch['used_mask'],total_pos_features, opt, negs=False)
            #     # unsup_cr_loss_neg = self.unsup_cr_loss(all_features, output_fm, out_mask,batch['used_mask'], total_pos_features, opt)
                unsup_cr_loss = unsup_cr_loss_pos 
                cr_loss += 0.1*unsup_cr_loss
            # print('out_fm', output_fm.shape)
            # out_fm = output_fm.view(1, 32, -1)
            # out_fm = out_fm.squeeze()
            # # print('pos_features', pos_features.shape)
            # pos_features = pos_features[:30, :]
            # sim_mat = torch.matmul(out_fm.T, pos_features.T)
            # sim_mat = sim_mat.mean(1, keepdim=True)
            # print('sim_mat', sim_mat)
            # torch.cuda.empty_cache()
            # sim_mat = sim_mat.view(output['hm'].shape[2:])
            # print('sim_mat', sim_mat.shape)
            # print('out_fm', out_fm.shape)
            # soft_negs = batch['soft_neg']
            # hard_pairs = self.miner(all_features, all_labels)
            # cr_loss += self.cr_loss(all_features, all_labels, hard_pairs)
            
            # print('cr_loss', cr_loss)
            # output.update({'sim_map': sim_mat})
        # if opt.wh_weight > 0:
        #     wh_loss += self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh']) / opt.num_stacks 

        # if opt.reg_offset and opt.off_weight > 0:
        #     off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg']) / opt.num_stacks
        # else:
        #   off_loss = torch.tensor(off_loss).float()
        if epoch > 0:

            if phase == 'train':
                loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + 0.5*cr_loss
            else:
                cr_loss = hm_loss * 0
                unsup_cr_loss = hm_loss * 0
                loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        # if epoch <= 35:
        #     if phase == 'train':
        #         unsup_cr_loss = hm_loss * 0
        #         loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + 0.1*cr_loss
        #     else:
        #         cr_loss = hm_loss * 0
        #         unsup_cr_loss = hm_loss * 0
        #         loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        # # else:
        #     if phase == 'train':
        #         loss = cr_loss
        #         unsup_cr_loss = hm_loss * 0
        #         hm_loss = hm_loss * 0
        #         wh_loss = wh_loss * 0
        #         off_loss = off_loss * 0
        #     else:
        #         loss = hm_loss
        #         cr_loss = hm_loss * 0
        #         unsup_cr_loss = hm_loss * 0
        #         hm_loss = hm_loss * 0
        #         wh_loss = wh_loss * 0
        #         off_loss = off_loss * 0


        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'cr_loss': cr_loss}

        return loss, loss_stats

class TomoTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(TomoTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'cr_loss']
        loss = TomoLoss(opt)
        return loss_states, loss 

    def debug(self, batch, output, iter_id):
        opt = self.opt 
        # reg = output['reg'] if not opt.reg_offset else None 
        # print('output sim mat', output['sim_map'].shape)
        dets = tomo_decode(output['hm'], reg=None)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:,:,:2] *= opt.down_ratio 
        post_dets = tomo_post_process(dets)
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, 3)
        name = batch['meta']['name']
        dets_gt[:,:,:2] *= opt.down_ratio 
        post_dets_gt = tomo_post_process(dets_gt)
        # print('name', name)
        # print(dets_gt.shape)
        # print('post_dets', post_dets)
        # print('post_dets_gt', post_dets_gt)
        
        for i in range(1):
            debugger = Debugger(dataset = opt.dataset, down_ratio = opt.down_ratio)
            tomo = batch['input'][i].detach().cpu().numpy()
            gts = post_dets_gt[i]
            preds = post_dets[i]
            name = name[i]
            # print('name', name)
            # print('preds', preds)
            det_zs = preds.keys()
            debugger.save_detection(preds, path = opt.debug_dir, name=name)
            # print(det_zs)
            for z in np.arange(50,80):
                out_z = output['hm'][i].detach().cpu().numpy()[0][z]
                out_z_n = output['hm'][i]
                # sim_mat_z = output['sim_map'].detach().cpu().numpy()[z]
                # print('out_z,', np.max(out_z))
                # sim_mat_z = cv2.normalize(sim_mat_z, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # print('out_z_n', out_z_n.shape)
                out_z_nms = _nms(out_z_n)
                # sim_mat_z_nms = _nms(output['sim_map'].unsqueeze(0))
                # print('sim_mat_z_nms', sim_mat_z_nms.shape)
                out_z_nms = out_z_nms.detach().cpu().numpy()[0][z]
                out_z_gt = batch['hm'][i].detach().cpu().numpy()[z]
                # print('out_z_gt', out_z_gt.shape)
                # out_z_n = out_z_n.detach().cpu().numpy()[0][z]
                out_z = np.expand_dims(out_z, 0)
                out_z_gt = np.expand_dims(out_z_gt, 0)
                # sim_mat_z = np.expand_dims(sim_mat_z, 0)
                pred = debugger.gen_colormap(out_z)
                gt = debugger.gen_colormap(out_z_gt)
                # sim_mat_c = debugger.gen_colormap(sim_mat_z)
                # gt = debugger
                tomo_z = cv2.normalize(tomo[z], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                tomo_z = np.clip(tomo_z * 255., 0, 255).astype(np.uint8)
                # print(np.min(tomo_z), np.max(tomo_z))
                img_slice = np.dstack((tomo_z, tomo_z, tomo_z))
                # debugger.add_blend_img(img_slice, pred, 'pred_hm')
                debugger.add_slice(pred, 'pred_hm')
                # debugger.add_slice(sim_mat_c, 'similarity_matrix')
                # debugger.add_slice(gt, 'gt_hm')
                debugger.add_blend_img(img_slice, gt, 'gt_hm')
                debugger.add_slice(img_slice, img_id = 'pred_out')
                debugger.add_slice(img_slice, img_id = 'gt_out')
                
                if z in preds.keys():
                    slice_coords = preds[z]
                # slice_coords = preds[z]
                    # print('slice_coords', slice_coords)
                    debugger.add_particle_detection(slice_coords, 8, img_id = 'pred_out')
                if z in gts.keys():
                    slice_coords = gts[z]
                    debugger.add_particle_detection(slice_coords, 8, img_id = 'gt_out')
                if opt.debug == 4:
                    debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id), slice_num = z)

                # print('color_map_gt', gt.shape)
                # print(out_z.shape)
                # print('gts', gts)
            # print('tomo,', tomo.shape)


            # print('dets', dets)
        # print(dets.shape)
        # print(dets_gt.shape)

    def save_results(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None 
        dets = tomo_decode(output['hm'], output['wh'], reg=reg, K = self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        return dets 
