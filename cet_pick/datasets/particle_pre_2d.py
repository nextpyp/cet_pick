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
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr, drop_out, center_out,swap_out, change_view
import math
from utils.lie_tools import random_SO3, constrained_SO3
from utils.project3d import Projector

class ParticlePre2DDataset(data.Dataset):
	
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


	def _convert_coord1d_to_3d(self, coord, width, height):

		z = coord // (width * height)
		x = (coord - z * width * height) % width 
		y = (coord - z * width * height - x) // width

		return [x,y,z]

	def __getitem__(self, index):
		sub_vol = self.subvols[index]
		zz = self.size[0]//2 
		sub_vol_c = np.zeros(sub_vol.shape)
		sub_vol_c[zz-4:zz+4] = sub_vol[zz-4:zz+4]
		projector = Projector(sub_vol_c)
		d_prob1 = np.random.random()
		if d_prob1 > 0.5:
			rots1 = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
			rots1 = rots1.unsqueeze(0).float()
		else:
			rots1 = constrained_SO3()
		rots2 = constrained_SO3()
		proj1 = projector.project(rots1)
		proj2 = projector.project(rots2)
		
		aug_1 = self.transforms(proj1)
		aug_2 = self.transforms(proj2)
	


		ret = {'input': aug_1.float(), 'input_aug': aug_2.float()}
		return ret

