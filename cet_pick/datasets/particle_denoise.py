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
import torchvision.transforms.functional as F

class ParticleDenoiseDataset(data.Dataset):
	



	def __getitem__(self, index):
		if self.split == 'train':
			# tomo = self.tomos[index]
			tomo = self.single_tomos[index]
			# print('noisy_in', tomo.size)
			noisy_in = self.transforms(tomo)
			# print('noisy_in after transform', noisy_in.shape)
			meta = {'shape': noisy_in.shape[1:]}
			# print('paried_tomo', paired_tomo)
			# print('paried_hm', paired_hm)
			# ret = {'input': img.astype(np.float32), 'input_aug': aug_img.astype(np.float32), 'hm': hm, 'hm_aug': hm_aug,'reg_mask': reg_mask, 'ind': ind, 'class': clas_z, 'gt_det': gt_det}
			# ret ={'input':cropped_tomo.astype(np.float32),'input_aug': cropped_tomo_aug.astype(np.float32), 'hm': cropped_hm, 'hm_aug': cropped_hm_aug, 'ind': ind, 'gt_det': gt_det, 'flip_prob': flip_prob, 'pair_inp':paired_tomo.astype(np.float32), 'pair_hm': paired_hm, 'paried_inp_aug': paired_tomo_aug.astype(np.float32), 'paired_hm_aug': paired_hm_aug}
			ret = {'noisy_in': noisy_in.float(), 'gt': noisy_in.float(), 'meta': meta}
			# ret ={'input':paired_tomo.astype(np.float32),'input_aug': paired_tomo_aug.astype(np.float32), 'hm': paired_hm, 'hm_aug': paired_hm_aug, 'ind': ind, 'gt_det': gt_det, 'flip_prob': flip_prob, 'pair_inp':paired_tomo.astype(np.float32), 'pair_hm': paired_hm, 'paried_inp_aug': paired_tomo_aug.astype(np.float32), 'paired_hm_aug': paired_hm_aug}
		else:
			tomo = self.single_tomos[index]

			name = self.tomo_names[index]
			ind = self.tomo_list_inds[index]
			if self.split == 'val':
				noisy_in = self.to_tensor(tomo)[:,:1024,:1024]
			else:
				noisy_in = self.to_tensor(tomo)
			meta = {'shape': noisy_in.shape[1:], 'name': name, 'ind': ind}
			img_size = noisy_in.shape
			h, w = img_size[1], img_size[2]
			size = max(h, w)
			need_pad = True
			if size == h and size == w:
				need_pad = False 
			if need_pad:
				left, top = 256, 256
				right = size - w + 256  
				bottom = size - h + 256
				# pad_mat = (top, bottom, left, right)
				pad_mat = (left, top, right, bottom)
				noisy_in_pad = F.pad(noisy_in, pad_mat, padding_mode= "reflect")

				ret = {'noisy_in': noisy_in_pad.float(), 'gt': noisy_in.float(), 'meta': meta}
			else:
				ret = {'noisy_in': noisy_in.float(), 'gt': noisy_in.float(), 'meta': meta}

		return ret

	