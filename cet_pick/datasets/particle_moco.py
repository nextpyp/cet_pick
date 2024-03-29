from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
import random
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr
import math
from random import choice 

class ParticleMocoDataset(data.Dataset):
	
	def _flip_coord_ud(self,ann,h):
		x, y, z = ann[0], ann[1], ann[2]
		new_y = h - y - 1
		return[x, new_y, z]

	def _flip_coord_lr(self,ann, h):
		x, y, z = ann[0], ann[1], ann[2]
		new_x = h - x - 1
		return[new_x, y, z]

	def _downscale_coord(self,ann):
		x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]
		return [x,y,z]




	def __getitem__(self, index):
		if self.split == 'train':
			# tomo = self.tomos[index]
			curr_ann = self.all_anns[index]
			# print('curr ann', curr_ann)
			tomo_ind = curr_ann[-1]
			tomo = self.tomos[tomo_ind]
			hm = self.hms[tomo_ind]
			ind = self.inds[tomo_ind]
			gt_det = self.gt_dets[tomo_ind]
			name = self.names[tomo_ind]
			# hm = self.hms[index]
			# ind = self.inds[index]
			# gt_det = self.gt_dets[index]

			flip_prob = np.random.random()
			# rot_time = np.random.randint()

			# name = self.names[index]
			depth, height, width = tomo.shape[0], tomo.shape[2], tomo.shape[1]
			# num_of_samples = gt_det.shape[0]
			# locs = np.random.randint(0, num_of_samples, 2)
			# loc = locs[0]
			# loc = curr_ann[:3]
			# print('loc', loc)
			offs_x_y = [-4,-3,-2, -1, 0, 1, 2,3,4]
			offs_z = [-1, 0, 1]
			off_x = random.sample(offs_x_y, 1)[0]
			off_y = random.sample(offs_x_y, 1)[0]
			off_z = random.sample(offs_z, 1)[0]
			# sampled_z_center = list(np.arange(5, depth-5))
			# sampled_x_center = list(np.arange(17, height//2 - 17))
			# sampled_y_center = list(np.arange(17, height//2 - 17))
			
			p = np.random.rand()
			rot = np.random.randint(0, 4)
			if self.opt.pn:
				if p <= 1:
					selected = choice([i for i in range(self.num_samples) if i not in [index]]) 
					selected_ann = self.all_anns[selected]
					tomo_ind_p = selected_ann[-1]
					tomo_p = self.tomos[tomo_ind_p]
					hm_p = self.hms[tomo_ind_p]
					ind_p = self.inds[tomo_ind_p]
					gt_det_p = self.gt_dets[tomo_ind_p]
					name_p = self.names[tomo_ind_p]

					x_c_r, y_c_r, z_c_r = 0,0,0
					# off_x_r, off_y_r, off_z_r = 0,0,0
					off_x_r = np.random.randint(0, 512)
					off_y_r = np.random.randint(0, 512)
					off_z_r = np.random.randint(0, 128)
				else:
					selected = choice([i for i in range(self.num_samples) if i not in [index]]) 
					selected_ann = self.all_anns[selected]

					tomo_ind_p = selected_ann[-1]
					tomo_p = self.tomos[tomo_ind_p]
					hm_p = self.hms[tomo_ind_p]
					ind_p = self.inds[tomo_ind_p]
					gt_det_p = self.gt_dets[tomo_ind_p]
					name_p = self.names[tomo_ind_p]
					x_c_r, y_c_r, z_c_r = np.random.randint(32, height-32), np.random.randint(32, height-32), np.random.randint(5, depth-5)
					off_x_r = np.random.randint(-32, 32)
					off_y_r = np.random.randint(-32, 32)
					off_z_r = np.random.randint(-5, 5)
			else:
				if p <= 0.9:
					selected = choice([i for i in range(self.num_samples) if i not in [index]]) 
					selected_ann = self.all_anns[selected]
					tomo_ind_p = selected_ann[-1]
					tomo_p = self.tomos[tomo_ind_p]
					hm_p = self.hms[tomo_ind_p]
					ind_p = self.inds[tomo_ind_p]
					gt_det_p = self.gt_dets[tomo_ind_p]
					name_p = self.names[tomo_ind_p]
					x_c_r, y_c_r, z_c_r = selected_ann[:3]
					# off_x_r, off_y_r, off_z_r = 0,0,0
					off_x_r = np.random.randint(-5, 5)
					off_y_r = np.random.randint(-5, 5)
					off_z_r = np.random.randint(-2, 2)
					# x_c_r, y_c_r, z_c_r = gt_det[locs[1]]
				else:
					selected = choice([i for i in range(self.num_samples) if i not in [index]]) 
					selected_ann = self.all_anns[selected]
					tomo_ind_p = selected_ann[-1]
					tomo_p = self.tomos[tomo_ind_p]
					hm_p = self.hms[tomo_ind_p]
					ind_p = self.inds[tomo_ind_p]
					gt_det_p = self.gt_dets[tomo_ind_p]
					name_p = self.names[tomo_ind_p]
					x_c_r, y_c_r, z_c_r = selected_ann[:3]
					off_x_r = np.random.randint(-32, 32)
					off_y_r = np.random.randint(-32, 32)
					off_z_r = np.random.randint(-5, 5)

				# x_c_r = random.sample(sampled_x_center, 1)[0]
				# y_c_r = random.sample(sampled_y_center, 1)[0]
				# z_c_r = random.sample(sampled_z_center, 1)[0]

			x_c_r, y_c_r, z_c_r = x_c_r + off_x_r, y_c_r + off_y_r, z_c_r + off_z_r
			x_c_r, y_c_r, z_c_r = np.clip(x_c_r, 17, height//2 - 17), np.clip(y_c_r, 17, width//2 - 17), np.clip(z_c_r, 3, depth - 3)
			x_c_r, y_c_r, z_c_r = int(x_c_r), int(y_c_r), int(z_c_r)
			
			# x_c, y_c, z_c = gt_det[loc]
			x_c, y_c, z_c = curr_ann[:3]
			x_c, y_c, z_c = x_c + off_x, y_c + off_y, z_c 
			x_c, y_c, z_c = np.clip(x_c, 17, height//2 - 17), np.clip(y_c, 17, width//2 - 17), np.clip(z_c, 3, depth - 3)
			x_c, y_c, z_c = int(x_c), int(y_c), int(z_c)

			up_xc, up_yc = int(x_c*self.opt.down_ratio), int(y_c*self.opt.down_ratio)
			up_xc_r, up_yc_r = int(x_c_r * self.opt.down_ratio), int(y_c_r * self.opt.down_ratio)

			# if p <= 0.5:
			# cropped_tomo_c = tomo_p[z_c_r-1:z_c_r+2, up_yc_r-32:up_yc_r+32, up_xc_r-32:up_xc_r+32]
			# cropped_hm_c = hm_p[z_c_r-1:z_c_r+2, y_c_r-16:y_c_r+16, x_c_r-16:x_c_r+16]
			cropped_tomo_c = tomo_p[z_c_r-3:z_c_r+3, up_yc_r-32:up_yc_r+32, up_xc_r-32:up_xc_r+32]
			cropped_hm_c = hm_p[z_c_r-3:z_c_r+3, y_c_r-16:y_c_r+16, x_c_r-16:x_c_r+16]
			# print('cropped_hm-c', cropped_hm_c)
			# else:
			# 	cropped_tomo_c = tomo_p[z_c_r-1:z_c_r+2, up_yc_r-32:up_yc_r+32, up_xc_r-32:up_xc_r+32]
			# 	cropped_hm_c = hm[z_c_r-1:z_c_r+2, y_c_r-16:y_c_r+16, x_c_r-16:x_c_r+16]

			# cropped_tomo = tomo[z_c-1:z_c+2, up_yc-32:up_yc+32, up_xc-32:up_xc+32]
			# cropped_hm = hm[z_c-1:z_c+2, y_c-16:y_c+16, x_c-16:x_c+16]
			cropped_tomo = tomo[z_c-3:z_c+3, up_yc-32:up_yc+32, up_xc-32:up_xc+32]
			cropped_hm = hm[z_c-3:z_c+3, y_c-16:y_c+16, x_c-16:x_c+16]
			# print('cropped_hm', cropped_hm)
			expand_tomo = np.expand_dims(cropped_tomo, axis=0)
			expand_hm = np.expand_dims(cropped_hm, axis=0)
			expand_tomo_c = np.expand_dims(cropped_tomo_c, axis=0)
			expand_hm_c = np.expand_dims(cropped_hm_c, axis=0)
			paired_tomo = np.concatenate((expand_tomo, expand_tomo_c), axis=0)
			paired_hm = np.concatenate((expand_hm, expand_hm_c), axis=0)
			if flip_prob <= 0.5:
				cropped_tomo_aug = flip_lr(cropped_tomo)
				cropped_hm_aug = flip_lr(cropped_hm)
				cropped_tomo_c_aug = flip_lr(cropped_tomo_c)
				cropped_hm_c_aug = flip_lr(cropped_hm_c)
			if flip_prob > 0.5:
				cropped_tomo_aug = flip_ud(cropped_tomo)
				cropped_hm_aug = flip_ud(cropped_hm)
				cropped_tomo_c_aug = flip_ud(cropped_tomo_c)
				cropped_hm_c_aug = flip_ud(cropped_hm_c)
			# cropped_tomo_aug = np.rot90(cropped_tomo_aug, k=rot, axes=(1,2))
			# cropped_hm_aug = np.rot90(cropped_hm_aug, k=rot, axes=(1,2))
			# cropped_tomo_c_aug = np.rot90(cropped_tomo_c_aug, k=rot, axes=(1,2))
			# cropped_hm_c_aug = np.rot90(cropped_hm_c_aug, k=rot, axes=(1,2))
			expand_tomo_aug = np.expand_dims(cropped_tomo_aug, axis=0)
			expand_hm_aug = np.expand_dims(cropped_hm_aug, axis=0)
			expand_tomo_c_aug = np.expand_dims(cropped_tomo_c_aug, axis=0)
			expand_hm_c_aug = np.expand_dims(cropped_hm_c_aug, axis=0)
			paired_tomo_aug = np.concatenate((expand_tomo_aug, expand_tomo_c_aug), axis=0)
			paired_hm_aug = np.concatenate((expand_hm_aug, expand_hm_c_aug), axis=0)
			# ret = {'input': img.astype(np.float32), 'input_aug': aug_img.astype(np.float32), 'hm': hm, 'hm_aug': hm_aug,'reg_mask': reg_mask, 'ind': ind, 'class': clas_z, 'gt_det': gt_det}
			# ret ={'input':cropped_tomo.astype(np.float32),'input_aug': cropped_tomo_aug.astype(np.float32), 'hm': cropped_hm, 'hm_aug': cropped_hm_aug, 'ind': ind, 'gt_det': gt_det, 'flip_prob': flip_prob, 'pair_inp':paired_tomo.astype(np.float32), 'pair_hm': paired_hm, 'paried_inp_aug': paired_tomo_aug.astype(np.float32), 'paired_hm_aug': paired_hm_aug}
			ret ={'input':paired_tomo.astype(np.float32),'input_aug': paired_tomo_aug.astype(np.float32), 'hm': paired_hm, 'hm_aug': paired_hm_aug, 'ind': ind, 'gt_det': gt_det, 'flip_prob': flip_prob, 'pair_inp':paired_tomo.astype(np.float32), 'pair_hm': paired_hm, 'paried_inp_aug': paired_tomo_aug.astype(np.float32), 'paired_hm_aug': paired_hm_aug,'rot': rot}
		else:
			tomo = self.tomos[index]
			hm = self.hms[index]
			ind = self.inds[index]
			gt_det = self.gt_dets[index]
			name = self.names[index]
			if tomo.shape[0] >=100 and tomo.shape[1] > 512:
				tomo_sec = tomo[:110, 200:700, 200:700]
				hm_sec = hm[:110, 100:350, 100:350]
				ret = {'input': tomo_sec.astype(np.float32), 'hm': hm_sec, 'ind': ind, 'gt_det': gt_det}
			else:
				ret ={'input': tomo.astype(np.float32), 'hm': hm, 'ind': ind, 'gt_det': gt_det}
			
		# if self.opt.contrastive:
		# 	ret.update({'soft_neg': soft_negs})



		# if self.opt.reg_offset:
		# 	ret.update({'reg':reg})
		if self.opt.debug > 0:
			
			meta = {'gt_det':gt_det, 'name': name}
			ret['meta'] = meta 
		return ret

		