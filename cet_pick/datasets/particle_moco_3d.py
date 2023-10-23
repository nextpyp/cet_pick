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

class ParticleMocoDataset3D(data.Dataset):
	
	def _flip_coord_ud(self,ann,h):
		x, y, z = ann[0], ann[1], ann[2]
		new_y = h - y - 1
		return[x, new_y, z]

	def _flip_coord_lr(self,ann, h):
		x, y, z = ann[0], ann[1], ann[2]
		new_x = h - x - 1
		return[new_x, y, z]

	def _downscale_coord(self,ann):
		x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]//self.opt.down_ratio
		return [x,y,z]

	# def _upscale_coord(self, ann):



	def __getitem__(self, index):
		if self.split == 'train':
			# print('3dd3d3d')
			tomo = self.tomos[index]
			hm = self.hms[index]
			ind = self.inds[index]
			gt_det = self.gt_dets[index]
			# print('index', index)
			# print('names', self.names)
			# print('tomo', tomo.shape)
			flip_prob = np.random.random()
			# print('gt_det', gt_det.shape)
			# print('gt_det x min', np.min(gt_det[:,0]))
			# print('gt_det y_min', np.min(gt_det[:,1]))
			name = self.names[index]
			depth, height, width = tomo.shape[0], tomo.shape[2], tomo.shape[1]
			num_of_samples = gt_det.shape[0]
			loc = random.randint(0, num_of_samples-1)
			# print('loc', loc)
			# print('height')
			sampled_z_center = list(np.arange(10, depth-10))
			sampled_x_center = list(np.arange(17, height//2 - 17))
			# # print('sampled_x_center', sampled_x_center)
			sampled_y_center = list(np.arange(17, height//2 - 17))
			x_c_r = random.sample(sampled_x_center, 1)[0]
			y_c_r = random.sample(sampled_y_center, 1)[0]
			z_c_r = random.sample(sampled_z_center, 1)[0]
			offs_x_y = [-2, -1, 0, 1, 2]
					# offs_z = [-2, -3, 3, 2]
			offs_z = [-1, 0, 1]
			off_x = random.sample(offs_x_y, 1)[0]
			off_y = random.sample(offs_x_y, 1)[0]
			off_z = random.sample(offs_z, 1)[0]
			x_c, y_c, z_c = gt_det[loc]
			x_c, y_c, z_c = x_c + off_x, y_c + off_y, z_c + off_z
			x_c, y_c, z_c = np.clip(x_c, 17, height//2 - 17), np.clip(y_c, 17, height//2 - 17), np.clip(z_c, 3, depth - 3)
			x_c, y_c, z_c = int(x_c), int(y_c), int(z_c)
			# print('x_c, y_c, z_c', x_c, y_c, z_c)
			up_xc, up_yc, up_zc = int(x_c*self.opt.down_ratio), int(y_c*self.opt.down_ratio), int(z_c*self.opt.down_ratio)
			up_xc_r, up_yc_r = int(x_c_r * self.opt.down_ratio), int(y_c_r * self.opt.down_ratio)
			# sampled_z_center = list(np.arange(6, depth-6))
			# sampled_x_center = list(np.arange(33, height - 33))
			# sampled_y_center = list(np.arange(33, height - 33))
			# x_c = random.sample(negative_region_x, 1)[0]
			# y_c = random.sample(negative_region_y, 1)[0]
			# z_c = random.sample(negative_region_z, 1)[0]
			cropped_tomo_c = tomo[z_c_r-2:z_c_r+2, up_yc_r-32:up_yc_r+32, up_xc_r-32:up_xc_r+32]
			cropped_hm_c = hm[z_c_r-1:z_c_r+1, y_c_r-16:y_c_r+16, x_c_r-16:x_c_r+16]
			# print('gt_det', gt_det)
			# print('zc', z_c)
			cropped_tomo = tomo[z_c-2:z_c+2, up_yc-32:up_yc+32, up_xc-32:up_xc+32]
			cropped_hm = hm[z_c-1:z_c+1, y_c-16:y_c+16, x_c-16:x_c+16]
			
			if flip_prob <= 0.5:
				cropped_tomo_aug = flip_lr(cropped_tomo)
				cropped_hm_aug = flip_lr(cropped_hm)
			if flip_prob > 0.5:
				cropped_tomo_aug = flip_ud(cropped_tomo)
				cropped_hm_aug = flip_ud(cropped_hm)


			# ret = {'input': img.astype(np.float32), 'input_aug': aug_img.astype(np.float32), 'hm': hm, 'hm_aug': hm_aug,'reg_mask': reg_mask, 'ind': ind, 'class': clas_z, 'gt_det': gt_det}
			ret ={'input':cropped_tomo.astype(np.float32),'input_aug': cropped_tomo_aug.astype(np.float32), 'hm': cropped_hm, 'hm_aug': cropped_hm_aug, 'ind': ind, 'gt_det': gt_det, 'flip_prob': flip_prob}
		else:
			tomo = self.tomos[index]
			hm = self.hms[index]
			ind = self.inds[index]
			gt_det = self.gt_dets[index]
			name = self.names[index]
			ret ={'input': tomo.astype(np.float32), 'hm': hm, 'ind': ind, 'gt_det': gt_det}

		# if self.opt.contrastive:
		# 	ret.update({'soft_neg': soft_negs})



		# if self.opt.reg_offset:
		# 	ret.update({'reg':reg})
		if self.opt.debug > 0:
			
			meta = {'gt_det':gt_det, 'name': name}
			ret['meta'] = meta 
		return ret


			
		






