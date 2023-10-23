from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import mrcfile
import cv2
import json
import os
import pandas as pd 
import numpy as np 
import math 

import torch.utils.data as data
from cet_pick.utils.loader import load_tomos_from_list
from cet_pick.utils.coordinates import match_coordinates_to_images
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr

class TOMOMoco3D(Dataset):
	num_classes = 1 
	default_resolution = [256, 256]

	def __init__(self, opt, split):
		super(TOMOMoco3D, self).__init__()
		if split == 'train':
			self.data_dir = os.path.join(opt.data_dir, opt.train_img_txt)
			self.coord_dir = os.path.join(opt.data_dir, opt.train_coord_txt)
			
		elif split == 'val':
			self.data_dir = os.path.join(opt.data_dir, opt.val_img_txt)
			self.coord_dir = os.path.join(opt.data_dir, opt.val_coord_txt)

		else:
			self.data_dir = os.path.join(opt.data_dir, opt.test_img_txt)
			self.coord_dir = os.path.join(opt.data_dir, opt.test_coord_txt)

		self.split = split 
		self.opt = opt 

		# self.images, self.targets, self.names = self.load_data()
		self.tomos, self.hms, self.inds, self.gt_dets, self.names = self.load_data()
		self.num_samples = len(self.tomos)
		# print('hello tomo')
		print('Loaded {} {} samples'.format(split, self.num_samples))
		

	def __len__(self):
		return self.num_samples

	def _downscale_coord(self,ann):
		x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]//self.opt.down_ratio
		return [x,y,z]

	def match_images_targets(self, images, targets):
		matched = match_coordinates_to_images(targets, images)
		tomos = []
		targets = []
		names = []
		for key in matched:
			names.append(key)
			tomos.append(matched[key]['tomo'])
			targets.append(matched[key]['coord'])
		return tomos, targets, names

	def load_data(self, image_ext=''):

		image_list = pd.read_csv(self.data_dir, sep='\t')
		coords = pd.read_csv(self.coord_dir, sep='\t')
		images = load_tomos_from_list(image_list.image_name, image_list.path)
		num_particles = len(coords)
		ims, targets, names = self.match_images_targets(images, coords)
		tomo = ims[0]
		coords = targets[0]
		depth, height, width = tomo.shape[0], tomo.shape[1], tomo.shape[2]
		num_objs = len(coords)
		output_h, output_w, output_depth = height // self.opt.down_ratio, width // self.opt.down_ratio, depth // self.opt.down_ratio
		num_class = self.num_classes
		hm = np.zeros((output_depth, output_h, output_w), dtype=np.float32) - 1
		ind = np.zeros((num_objs), dtype=np.int64)
		draw_gaussian = draw_msra_gaussian_3d if self.opt.mse_loss else draw_umich_gaussian_3d
		gt_det = []
		h = self.opt.bbox // self.opt.down_ratio
		for k in range(num_objs):
			ann = coords[k]
			radius = gaussian_radius((math.ceil(h), math.ceil(h)))
			radius = max(0, int(radius))
			ann = self._downscale_coord(ann)
			ann = np.array(ann)
			ct_int = ann.astype(np.int32)
			z_coord = ct_int[-1]
			draw_gaussian(hm, ct_int, radius,  0, 0, 0, discrete=False)
			ind[k] = ct_int[2] * (output_w * output_h) + ct_int[1] * output_w + ct_int[0]
			gt_det.append(ann)
		gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
		tomos = [tomo]
		hms = [hm]
		inds = [ind]
		# print('tomo', tomo.shape)
		# print('tomos', tomos)
		gt_dets = [gt_det]
		# print('names', names)
		return tomos, hms, inds, gt_dets, names