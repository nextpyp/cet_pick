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

class ContrastiveDataset(data.Dataset):

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
		# negative_region_x = random.randint(20, 490)
		negative_region_x = np.arange(20,490)
		negative_region_y = np.arange(20,490)
		negative_region_z = np.concatenate((np.arange(-10, -3), np.arange(3, 10)))
		h = self.opt.bbox 
		pos_patches = np.zeros((16, 5, h, w))
		neg_patches = np.zeros((16, 5, h, w))
		pos_indices = np.zeros((16, 3))
		neg_indices = np.zeros((16, 3))
		if self.split == 'train':
			flip_prob = np.random.random()
		else:
			flip_prob = 0.5
		selected_positive = random.sample(range(num_objs), 16)
		random.shuffle(negative_region_x)
		random.shuffle(negative_region_y)
		random.shuffle(negative_region_z)
		i = 0
		for k in selected_positive:
			ann = coords[k]
			ct_int = np.array(ann).astype(np.int32)
			x_coord, y_coord, z_coord = ct_int[0], ct_int[1], ct_int[-1]
			pos_indices[i] = ct_int
			cropped_patch = img[z_coord-2:z_coord+3, y_coord - int(h//2): y_coord + int(h//2), x_coord - int(h//2): x_coord + int(h//2)]
			pos_patches[i] = cropped_patch
			i += 1 
		for j in range(16):
			x_coord, y_coord, z_coord = negative_region_x[j], negative_region_y[j], negative_region_z[j]
			cropped_patch = img[z_coord-2:z_coord+3, y_coord - int(h//2): y_coord + int(h//2), x_coord - int(h//2): x_coord + int(h//2)]
			neg_patches[j] = cropped_patch
			neg_indices[j] = [x_coord, y_coord, z_coord]
		ret = {'pos_input': pos_patches.astype(np.float32), 'neg_input': neg_patches.astype(np.float32)}

		if self.opt.debug > 0:
			meta = {'pos_ind': pos_indices, 'neg_ind': neg_indices, 'name':name}
			ret['meta'] = meta 

		return ret




		# _flip_lr= False
		# _flip_ud = False
		# flip = False
		# if flip_prob <= 0.33:
		# 	img = flip_lr(img)
		# 	_flip_lr = True
		# 	flip = True 
		# elif flip_prob > 0.67:
		# 	img = flip_ud(img)
		# 	_flip_ud = True 
		# 	flip = True
		# else:
		# 	img = img
