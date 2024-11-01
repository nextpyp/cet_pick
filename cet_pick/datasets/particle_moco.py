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
			curr_ann = self.all_anns[index]
			tomo_ind = curr_ann[-1]
			tomo = self.tomos[tomo_ind]
			hm = self.hms[tomo_ind]
			ind = self.inds[tomo_ind]
			gt_det = self.gt_dets[tomo_ind]
			name = self.names[tomo_ind]

			translation_pixels = int(self.opt.bbox * self.opt.translation_ratio)

			flip_prob = np.random.random()
			depth, height, width = tomo.shape[0], tomo.shape[1], tomo.shape[2]
			offs_x_y = [-4,-3,-2, -1, 0, 1, 2,3,4]
			offs_z = [-1, 0, 1]
			off_x = random.sample(offs_x_y, 1)[0]
			off_y = random.sample(offs_x_y, 1)[0]
			off_z = random.sample(offs_z, 1)[0]
			p = np.random.rand()
			if self.opt.pn:
				if p <= 0.5:
					selected = choice([i for i in range(self.num_samples) if i not in [index]]) 
					selected_ann = self.all_anns[selected]
					tomo_ind_p = selected_ann[-1]
					tomo_p = self.tomos[tomo_ind_p]
					depth_p, height_p, width_p = tomo_p.shape[0], tomo_p.shape[1], tomo_p.shape[2]
					hm_p = self.hms[tomo_ind_p]
					ind_p = self.inds[tomo_ind_p]
					gt_det_p = self.gt_dets[tomo_ind_p]
					name_p = self.names[tomo_ind_p]

					x_c_r, y_c_r, z_c_r = 0,0,0
					off_x_r = np.random.randint(0, width_p)
					off_y_r = np.random.randint(0, height_p)
					off_z_r = np.random.randint(0, depth_p)
				else:
					selected = choice([i for i in range(self.num_samples) if i not in [index]]) 
					selected_ann = self.all_anns[selected]

					tomo_ind_p = selected_ann[-1]
					tomo_p = self.tomos[tomo_ind_p]
					depth_p, height_p, width_p = tomo_p.shape[0], tomo_p.shape[1], tomo_p.shape[2]
					hm_p = self.hms[tomo_ind_p]
					ind_p = self.inds[tomo_ind_p]
					gt_det_p = self.gt_dets[tomo_ind_p]
					name_p = self.names[tomo_ind_p]
					
					x_c_r, y_c_r, z_c_r = selected_ann[:3]
					off_x_r = np.random.randint(-1*translation_pixels, translation_pixels)
					off_y_r = np.random.randint(-1*translation_pixels, translation_pixels)
					off_z_r = np.random.randint(-5, 5)
			else:
				# x y z translation augmentation

				if p <= 0.8:
					selected = choice([i for i in range(self.num_samples) if i not in [index]]) 
					selected_ann = self.all_anns[selected]
					tomo_ind_p = selected_ann[-1]
					tomo_p = self.tomos[tomo_ind_p]
					depth_p, height_p, width_p = tomo_p.shape[0], tomo_p.shape[1], tomo_p.shape[2]
					hm_p = self.hms[tomo_ind_p]
					ind_p = self.inds[tomo_ind_p]
					gt_det_p = self.gt_dets[tomo_ind_p]
					name_p = self.names[tomo_ind_p]
					x_c_r, y_c_r, z_c_r = selected_ann[:3]
					off_x_r = np.random.randint(-5, 5)
					off_y_r = np.random.randint(-5, 5)
					off_z_r = np.random.randint(-2, 2)
				else:
					selected = choice([i for i in range(self.num_samples) if i not in [index]]) 
					selected_ann = self.all_anns[selected]
					tomo_ind_p = selected_ann[-1]
					tomo_p = self.tomos[tomo_ind_p]
					depth_p, height_p, width_p = tomo_p.shape[0], tomo_p.shape[1], tomo_p.shape[2]
					hm_p = self.hms[tomo_ind_p]
					ind_p = self.inds[tomo_ind_p]
					gt_det_p = self.gt_dets[tomo_ind_p]
					name_p = self.names[tomo_ind_p]
					x_c_r, y_c_r, z_c_r = selected_ann[:3]
					off_x_r = np.random.randint(-1*translation_pixels, translation_pixels)
					off_y_r = np.random.randint(-1*translation_pixels, translation_pixels)
					off_z_r = np.random.randint(-2, 2)

			x_c_r, y_c_r, z_c_r = x_c_r + off_x_r, y_c_r + off_y_r, z_c_r + off_z_r
			x_c_r, y_c_r, z_c_r = np.clip(x_c_r, 17, width_p//2 - 17), np.clip(y_c_r, 17, height_p//2 - 17), np.clip(z_c_r, 3, depth_p - 3)
			x_c_r, y_c_r, z_c_r = int(x_c_r), int(y_c_r), int(z_c_r)
			

			x_c, y_c, z_c = curr_ann[:3]
			x_c, y_c, z_c = x_c + off_x, y_c + off_y, z_c 
			x_c, y_c, z_c = np.clip(x_c, 17, width//2 - 17), np.clip(y_c, 17, height//2 - 17), np.clip(z_c, 3, depth - 3)
			x_c, y_c, z_c = int(x_c), int(y_c), int(z_c)

			up_xc, up_yc = int(x_c*self.opt.down_ratio), int(y_c*self.opt.down_ratio)
			up_xc_r, up_yc_r = int(x_c_r * self.opt.down_ratio), int(y_c_r * self.opt.down_ratio)

			# fixed input patch size of 64 * 64, downscale by 2 is 32 * 32
			cropped_tomo_c = tomo_p[z_c_r-3:z_c_r+3, up_yc_r-32:up_yc_r+32, up_xc_r-32:up_xc_r+32]
			cropped_hm_c = hm_p[z_c_r-3:z_c_r+3, y_c_r-16:y_c_r+16, x_c_r-16:x_c_r+16]

			cropped_tomo = tomo[z_c-3:z_c+3, up_yc-32:up_yc+32, up_xc-32:up_xc+32]
			cropped_hm = hm[z_c-3:z_c+3, y_c-16:y_c+16, x_c-16:x_c+16]

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

			expand_tomo_aug = np.expand_dims(cropped_tomo_aug, axis=0)
			expand_hm_aug = np.expand_dims(cropped_hm_aug, axis=0)
			expand_tomo_c_aug = np.expand_dims(cropped_tomo_c_aug, axis=0)
			expand_hm_c_aug = np.expand_dims(cropped_hm_c_aug, axis=0)
			paired_tomo_aug = np.concatenate((expand_tomo_aug, expand_tomo_c_aug), axis=0)
			paired_hm_aug = np.concatenate((expand_hm_aug, expand_hm_c_aug), axis=0)

			ret ={'input':paired_tomo.astype(np.float32),'input_aug': paired_tomo_aug.astype(np.float32), 'hm': paired_hm, 'hm_aug': paired_hm_aug, 'ind': ind, 'gt_det': gt_det, 'flip_prob': flip_prob, 'pair_inp':paired_tomo.astype(np.float32), 'pair_hm': paired_hm, 'paried_inp_aug': paired_tomo_aug.astype(np.float32), 'paired_hm_aug': paired_hm_aug}
		else:
			tomo = self.tomos[index]
			hm = self.hms[index]
			ind = self.inds[index]
			gt_det = self.gt_dets[index]
			name = self.names[index]

			# debug val only operates on a subregion of input tomo to avoid oom
			if tomo.shape[0] >=100 and tomo.shape[1] > 512:
				tomo_sec = tomo[:110, 200:700, 200:700]
				hm_sec = hm[:110, 100:350, 100:350]
				ret = {'input': tomo_sec.astype(np.float32), 'hm': hm_sec, 'ind': ind, 'gt_det': gt_det}
			else:
				ret ={'input': tomo.astype(np.float32), 'hm': hm, 'ind': ind, 'gt_det': gt_det}
			

		if self.opt.debug > 0:
			
			meta = {'gt_det':gt_det, 'name': name}
			ret['meta'] = meta 
		return ret

		