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

class TOMOMocoClass:
	num_classes = 1 
	default_resolution = [256, 256]

	def __init__(self, opt, split, width_xy, width_z):
		super(TOMOMocoClass, self).__init__()
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
		if self.split == 'train' or self.split == 'val':
			# self.tomos, self.hms, self.inds, self.gt_dets, self.names, self.all_anns = self.load_data()
			self.tomos, self.hms, self.inds, self.gt_dets, self.names = self.load_data()
			self.num_samples = len(self.tomos)
			im = self.tomos[0]
			d,w,h = im.shape 
			if split == 'train':
				self.n = d*w*h*self.num_samples
			else:
				self.n = self.num_samples
		else:
			self.names, self.paths, self.images = self.load_data()
			self.n = len(self.names)
			self.num_samples = self.n


		self.width_z = width_z
		self.width_xy = width_xy
		
		
		# print('hello tomo')
		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.n
		# num_sample = 128*256*256
		# return num_sample

	def _downscale_coord(self,ann, compress=False):
		if compress:
			x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]//2  
		else:
			x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]
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
		
		if self.split == 'train' or self.split == 'val':
			image_list = pd.read_csv(self.data_dir, sep='\t')
			coords = pd.read_csv(self.coord_dir, sep='\t')
			images = load_tomos_from_list(image_list.image_name, image_list.rec_path, order=self.opt.order, compress=self.opt.compress, denoise=self.opt.gauss)
			num_particles = len(coords)
			ims, targets, names = self.match_images_targets(images, coords)
			num_of_tomos = len(ims)
			# print('num_of_tomos', num_of_tomos)
			# print('coords', coords)
			tomos = []
			hms = []
			inds = []
			gt_dets = []
			for i in range(num_of_tomos):
				tomo = ims[i]
				coords = targets[i]
				depth, height, width = tomo.shape[0], tomo.shape[1], tomo.shape[2]
				num_objs = len(coords)
				# print('num_objs', num_objs)
				output_h, output_w = height // self.opt.down_ratio, width // self.opt.down_ratio
				num_class = self.num_classes
				# if self.opt.pn:
				# 	hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
				# else:
				hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
				ind = np.zeros((num_objs), dtype=np.int64)
				draw_gaussian = draw_umich_gaussian_3d
				gt_det = []
				# all_coords = [np.arange(output_h), np.arange(output_w), np.arange(width)]
				h = self.opt.bbox // self.opt.down_ratio
				for k in range(num_objs):
					ann = coords[k]
					radius = gaussian_radius((math.ceil(h), math.ceil(h)))
					radius = max(0, int(radius))
					ann = self._downscale_coord(ann,compress = self.opt.compress)
					ann = np.array(ann)
					ct_int = ann.astype(np.int32)
					z_coord = ct_int[-1]
					if self.opt.fiber:
						draw_gaussian(hm, ct_int, radius,  1, 0, 0.2, discrete=True)
					else:
						draw_gaussian(hm, ct_int, radius,  1, 0.1, 0.5, discrete=True)
						# draw_gaussian(hm, ct_int, radius,  0, 0, 0, discrete=False)
					ind[k] = ct_int[2] * (output_w * output_h) + ct_int[1] * output_w + ct_int[0]
					# print('ann', ann)
					gt_det.append(ann)
				gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
			# all_coords = np.array(all_coords)
				if self.split == 'train':
					if not self.opt.pn:
						hm[np.where(hm == 0)] = -1
				tomos.append(tomo)
				hms.append(hm)
				inds.append(ind)
				gt_dets.append(gt_det)
			# gt_dets = [gt_det]

			return tomos, hms, inds, gt_dets, names
		else:
			image_list = pd.read_csv(self.data_dir, sep='\t')
			names = image_list.image_name
			paths = image_list.rec_path
			ims = load_tomos_from_list(image_list.image_name, image_list.rec_path,order=self.opt.order, compress=self.opt.compress, denoise=self.opt.gauss)
			images = []
			for k, v in ims.items():
				images.append(v)
			return names, paths, images