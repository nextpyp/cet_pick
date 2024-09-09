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

class ParticleDataset(data.Dataset):
	
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
		img = self.images[index]
		coords = self.targets[index]
		name = self.names[index]
		depth, height, width = img.shape[0], img.shape[1], img.shape[2]
		num_objs = len(coords)

		output_h, output_w = height // self.opt.down_ratio, width // self.opt.down_ratio
		num_class = self.num_classes
		negative_region_x = list(np.arange(5,output_h - 5))
		negative_region_y = list(np.arange(5,output_w - 5))
		negative_region_z = list(np.arange(0, 10))
		if self.split == 'train':
			hm = np.zeros((depth, output_h, output_w), dtype=np.float32) - 1
		else:
			hm = np.zeros((depth, output_h, output_w), dtype=np.float32)

		used_mask = np.zeros((depth, output_h, output_w), dtype=np.float32)
		wh = np.zeros((num_objs, 1), dtype=np.float32)
		reg = np.zeros((num_objs, 2), dtype=np.float32)
		ind = np.zeros((num_objs), dtype=np.int64)
		clas_z = np.zeros((depth), dtype=np.float32)
		reg_mask = np.zeros((num_objs), dtype=np.uint8)
		draw_gaussian = draw_msra_gaussian_3d if self.opt.mse_loss else draw_umich_gaussian_3d

		gt_det = []
		soft_negs = []
		if self.split == 'train':
			flip_prob = np.random.random()
		else:
			flip_prob = 0.5
		_flip_lr= False
		_flip_ud = False
		flip = False
		if flip_prob <= 0.33:
			img = flip_lr(img)
			_flip_lr = True
			flip = True 
		elif flip_prob > 0.67:
			img = flip_ud(img)
			_flip_ud = True 
			flip = True
		else:
			img = img
		h = self.opt.bbox // self.opt.down_ratio
		for k in range(num_objs):
			ann = coords[k]
			if _flip_lr:
				ann = self._flip_coord_lr(ann, width)
			if _flip_ud:
				ann = self._flip_coord_ud(ann, height)
			radius = gaussian_radius((math.ceil(h), math.ceil(h)))
			radius = max(0, int(radius))
			radius = self.opt.hm_gauss if self.opt.mse_loss else radius
			ann = self._downscale_coord(ann)
			ann = np.array(ann)
			ct_int = ann.astype(np.int32)
			z_coord = ct_int[-1]
			clas_z[int(z_coord)] = 1
			draw_gaussian(hm, ct_int, radius, 0, 0, 0, discrete=False)
			wh[k] = h
			used_mask[z_coord, ct_int[1], ct_int[0]] = 1
			ind[k] = ct_int[2] * (output_w * output_h) + ct_int[1] * output_w + ct_int[0]
			reg[k] = ann[:2] - ct_int[:2]
			reg_mask[k] = 1
			gt_det.append(ann)
			if self.opt.contrastive:
				if self.split == 'train':
					offs_x_y = [-2, -1, 1, 2]
					offs_z = [-1, 1]
					off_x = random.sample(offs_x_y, 1)[0]
					off_y = random.sample(offs_x_y, 1)[0]
					off_z = random.sample(offs_z, 1)[0]
					off_coord = ann - np.array([off_x, off_y, off_z])
					if off_coord[-1] >= 128:
						off_coord[-1] = 127
					off_coord = np.clip(off_coord, 0, 255)
					used_mask[off_coord[-1], off_coord[-2], off_coord[-3]] = 1
					soft_negs.append(off_coord)
				else:
					soft_negs.append([0,0,0])
					used_mask[0,0,0] = 1


		gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
		if self.opt.contrastive:
			hard_negs = []
			num_of_hard_negs = 60
			for i in range(num_of_hard_negs):
				x_neg = random.sample(negative_region_x, 1)[0]
				y_neg = random.sample(negative_region_y, 1)[0]
				z_neg = random.sample(negative_region_z, 1)[0]
				neg_ann = np.array([x_neg, y_neg, z_neg])
				hm[z_neg, y_neg, x_neg] = 0
				used_mask[z_neg, y_neg, x_neg] = 1
				hard_negs.append(neg_ann)
			hard_negs = np.array(hard_negs, dtype=np.float32)
			soft_negs = np.array(soft_negs, dtype=np.float32)

		ind = np.random.randint(0, 128, 16)

		ret = {'input': img[ind].astype(np.float32), 'input_aug': img[ind].astype(np.float32), 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'class': clas_z[ind], 'gt_det': gt_det, 'used_mask':used_mask}
		if self.opt.contrastive:
			ret.update({'soft_neg': soft_negs, 'hard_neg': hard_negs})



		if self.opt.reg_offset:
			ret.update({'reg':reg})
		if self.opt.debug > 0:
			
			meta = {'gt_det':gt_det, 'name': name}
			ret['meta'] = meta 
		return ret






