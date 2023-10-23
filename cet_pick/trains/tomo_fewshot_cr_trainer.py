from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from cet_pick.models.loss import FocalLoss, RegL1Loss, RegLoss, SupConLoss, UnSupConLoss, SupConLossV2, SupConLossV2_more
from cet_pick.models.loss import KMeansVMFLoss, PartialSupLoss
from cet_pick.models.decode import tomo_decode
from cet_pick.models.decode import _nms
from cet_pick.models.utils import _sigmoid 
from cet_pick.trains.base_trainer import BaseTrainer 
from cet_pick.utils.debugger import Debugger
from cet_pick.utils.post_process import tomo_post_process, tomo_cluster_postprocess
from cet_pick.models.kmeans import MPKMeans


import cv2

from pytorch_metric_learning import miners, losses

class TomoKMLoss(torch.nn.Module):
    def __init__(self, opt):
        super(TomoKMLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        # self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else RegLoss() if opt.reg_loss == 'sl1' else None 
        # self.crit_wh = self.crit_reg 
        # self.cr_loss = SupConLossV2(0.07, 0.07, 0.07)
        # self.cr_loss = SupConLossV2(opt.temp, opt.temp, opt.temp)
        self.vmf_loss = KMeansVMFLoss(opt.temp)
        self.sup_loss = PartialSupLoss(opt.temp)
        self.kmeans = MPKMeans(opt, opt.n_clusters, exist_labels = 2, max_iter = 30, emb_dim = 16)
        
        # self.cr_loss = SupConLossV2_more(0.07,0.07,0.07)
        # self.cr_loss = losses.TripletMarginLoss()
        # self.unsup_cr_loss = UnSupConLoss(0.07)
        
        self.opt = opt 

    def forward(self, outputs, batch, epoch, phase, outputs_cr = None, cluster_center = None, cluster_ind = None):
        opt = self.opt 
        # hm_loss, wh_loss, off_loss, cr_loss = 0, 0, 0, 0
        # cr_loss, hm_loss = 0, 0
        vmf_loss, sup_loss = 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            # if outputs_cr is not None:
            #     output_cr = outputs_cr[s]
            # if not opt.mse_loss:

            #     output['hm'] = _sigmoid(output['hm'])
                # if outputs_cr is not None:
                #     output_cr['hm'] = _sigmoid(output_cr['hm'])
        #         out_mask = _sigmoid(7*(output['hm']-0.5))


        # hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks 
        # hm_loss += self.crit(output_cr['hm'], batch['hm_c']) / opt.num_stacks
        if opt.contrastive and phase == "train":
            if outputs_cr is not None:
                output_cr = outputs_cr[0]
                output_fm_cr = output_cr['proj']
                flip_prob = batch['flip_prob']
                print('output_fm_cr', output_fm_cr.shape)
                if flip_prob > 0.5:
                    output_fm_cr = output_fm_cr.flip(-2)
                else:
                    output_fm_cr = output_fm_cr.flip(-1)
                output_fm_cr = output_fm_cr.reshape((1, 16, -1))
                output_fm_cr = output_fm_cr.squeeze().T
            # hm_loss += self.crit(output_cr['hm'], batch['hm_c']) / opt.num_stacks
            output_fm = output['proj']
            # print('output_fm', output_fm.shape)
            # reshape the output projection to 1 * dim * num_pixels 
            output_fm = output_fm.reshape((1, 16, -1))
            output_fm = output_fm.squeeze().T 
            print('output_fm_shape', output_fm.shape)

            
            gt_labels = batch['lb_map']

            print('gt_labels', gt_labels.shape)
            gt_labels_f = gt_labels.reshape(1, -1)
            gt_labels_s = gt_labels_f.squeeze()
            print('gt_labels_f', gt_labels_s.shape)
            neg_pos = gt_labels_s.eq(2).float()
            neg_pos_bool = neg_pos.bool()
            pos_pos = gt_labels_s.eq(1).float()
            pos_pos_bool = pos_pos.bool()
            neg_pos_feat = output_fm[neg_pos_bool,:]
            neg_pos_feat_cr = output_fm_cr[neg_pos_bool, :]
            pos_pos_feat = output_fm[pos_pos_bool,:]
            pos_pos_feat_cr = output_fm_cr[pos_pos_bool,:]
            neg_cos = torch.matmul(neg_pos_feat, neg_pos_feat_cr.T)
            print('neg_cos', neg_cos)
            pos_cos = torch.matmul(pos_pos_feat, pos_pos_feat_cr.T)
            print('pos_cos', pos_cos)
            # neg_pos_t = gt_labels_s[neg_pos_bool]

            # print('neg_pos', neg_pos_t)
            ml_graph = batch['ml_graph']
            cl_graph = batch['cl_graph']
            if cluster_center is None:
                print('is none', epoch)
                cluster_centers, labels = self.kmeans.mpkmeans_with_initial_labels(output_fm, gt_labels, ml_graph, cl_graph)
            else:
                print('is not none', epoch)
                cluster_centers, labels = self.kmeans.mpkmeans_with_old_centers(output_fm, cluster_center, ml_graph, cl_graph)
                # cluster_centers, labels = self.kmeans.mpkmeans_with_initial_labels(output_fm, gt_labels, ml_graph, cl_graph)
            vmf_loss += self.vmf_loss(output_fm, labels, cluster_centers, self.opt)

            
            gt_labels_f = gt_labels_f.T.long()
            # gt_labels_f.squeeze().long()
            tru_index = gt_labels_f == 1
            # print('labels', labels.shape)
            # labels = labels.squeeze()
            actual_lb = labels[tru_index]
            # print('actual_lb', actual_lb)
            cluster_ind = torch.mode(actual_lb).values.item()

            sup_loss += self.sup_loss(output_fm, gt_labels_f, self.opt)
            # gt_hm = batch['hm']
            # gt_hm_cr = batch['hm_c']
            # output_fm_cr = output_cr['proj']
            # print('output_fm', output_fm.shape)
            # print('gt_hm', gt_hm.shape)
            # cr_loss += self.cr_loss(output_fm, gt_hm, opt)
            # cr_loss = hm_loss * 0
            # loss = cr_loss
            # loss =  hm_loss + cr_loss * 0.1
            # loss = hm_loss
            loss = vmf_loss + self.opt.cr_weight * sup_loss
            # loss = hm_loss + cr_loss * self.opt.cr_weight
            loss_stats = {'loss': loss,'vmf_loss': vmf_loss, 'sup_loss': sup_loss}
            return loss, loss_stats, cluster_centers, cluster_ind

        else:
            gt_labels = batch['hm']
            output_fm = output['proj']
            # reshape the output projection to 1 * dim * num_pixels 
            # output_fm = output_fm.reshape((1, 16, -1))
            # output_fm = output_fm.squeeze().T 
            cluster_used = cluster_center[cluster_ind]
            cluster_used.requires_grad = False
            print('cluster_used', cluster_used)
            print(cluster_used.shape)
            similarities = tomo_cluster_postprocess(cluster_used, output_fm)
            similarities = similarities.unsqueeze(0)
            # print('similarities', similarities.shape)
            # print(similarities)
            # cr_loss = hm_loss * 0
            # loss = hm_loss
            loss = self.crit(similarities, gt_labels)
            vmf_loss = loss * 0
            sup_loss = loss 

            loss_stats = {'loss': loss,'vmf_loss': vmf_loss, 'sup_loss': sup_loss}
            return loss, loss_stats, cluster_center, cluster_ind

        # loss_stats = {'loss': loss,'hm_loss': hm_loss, 'cr_loss': cr_loss}
        

        
class TomoKMTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(TomoKMTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss','vmf_loss','sup_loss']
        loss = TomoKMLoss(opt)
        return loss_states, loss 

    def debug(self, batch, output, clustercenter, iter_id, output_cr = None):
        opt = self.opt 
        # reg = output['reg'] if not opt.reg_offset else None 
        # print('output sim mat', output['sim_map'].shape)
        # dets = tomo_decode(output['hm'], reg=None)
        # if outputs_cr is not None:
        #         output_cr = outputs_cr[0]
        #         output_fm_cr = output_cr['proj']
        #         flip_prob = batch['flip_prob']
        #         print('output_fm_cr', output_fm_cr.shape)
        #         if flip_prob > 0.5:
        #             output_fm_cr = output_fm_cr.flip(-2)
        #         else:
        #             output_fm_cr = output_fm_cr.flip(-1)
        #         output_fm_cr = output_fm_cr.reshape((1, 16, -1))

        feature_maps = output['proj']
        print('feature_maps', feature_maps.shape)
        similarities = tomo_cluster_postprocess(clustercenter, feature_maps)
        similarities = similarities.unsqueeze(0).unsqueeze(0)
        print('similarities', similarities.shape)
        dets = tomo_decode(similarities, reg=None)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:,:,:2] *= opt.down_ratio 
        post_dets = tomo_post_process(dets, z_dim_tot = 128)
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, 3)
        name = batch['meta']['name']
        dets_gt[:,:,:2] *= opt.down_ratio 
        post_dets_gt = tomo_post_process(dets_gt)
        # print('name', name)
        # print(dets_gt.shape)
        # print('post_dets', post_dets)
        # print('post_dets_gt', post_dets_gt)
        print('tomo', batch['input'].shape)
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
                # out_z = output['hm'][i].detach().cpu().numpy()[0][z]
                # out_z_n = output['hm'][i]
                out_z = similarities[i].detach().cpu().numpy()[0][z]
                out_z_n = similarities[i]
                # print('out_proj_fist', output['proj'][i].shape)
                # out_z_sum = torch.sum(output['proj'][i], axis = 0)
                # out_z_sum = output['proj'][i][0]
                # print('out z sum shape', out_z_sum.shape)
                # # out_proj = output['proj'][i].detach().cpu().numpy()[0][z]
                # out_proj = out_z_sum.detach().cpu().numpy()[z]
                # print('out_proj', out_proj.shape)
                # sim_mat_z = output['sim_map'].detach().cpu().numpy()[z]
                # print('out_z,', np.max(out_z))
                # sim_mat_z = cv2.normalize(sim_mat_z, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # print('out_z_n', out_z_n.shape)
                out_z_nms = _nms(out_z_n)
                # sim_mat_z_nms = _nms(output['sim_map'].unsqueeze(0))
                # print('sim_mat_z_nms', sim_mat_z_nms.shape)
                out_z_nms = out_z_nms.detach().cpu().numpy()[0][z]
                out_z_gt = batch['lb_map'][i].detach().cpu().numpy()[z]
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
                # out_p = cv2.normalize(out_proj, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                tomo_z = np.clip(tomo_z * 255., 0, 255).astype(np.uint8)
                # out_p = np.clip(out_p * 255., 0, 255).astype(np.uint8)
                # proj_slice = np.dstack((out_p, out_p, out_p))
                # print(np.min(tomo_z), np.max(tomo_z))
                img_slice = np.dstack((tomo_z, tomo_z, tomo_z))
                # debugger.add_blend_img(img_slice, pred, 'pred_hm')
                debugger.add_slice(pred, 'pred_hm')
                # debugger.add_slice(sim_mat_c, 'similarity_matrix')
                # debugger.add_slice(gt, 'gt_hm')
                debugger.add_blend_img(img_slice, gt, 'gt_hm')

                debugger.add_slice(img_slice, img_id = 'pred_out')
                debugger.add_slice(img_slice, img_id = 'gt_out')
                # debugger.add_slice(proj_slice, img_id = 'project_features')
                
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

    def save_results(self, output, cluster_center, batch, results):
        # reg = output['reg'] if self.opt.reg_offset else None 
        feature_maps = output['proj']
        print('feature_maps', feature_maps.shape)
        similarities = tomo_cluster_postprocess(cluster_center, feature_maps)
        similarities = similarities.unsqueeze(0)
        dets = tomo_decode(similarities, reg=None)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        return dets 