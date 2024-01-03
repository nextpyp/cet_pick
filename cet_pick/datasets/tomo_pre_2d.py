from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import mrcfile
import cv2
import json
import os
import pandas as pd 
import torchvision.transforms as T
import numpy as np 
import math 
import torch
import torch.utils.data as data
from cet_pick.utils.loader import load_tomos_from_list, cutup
from cet_pick.utils.coordinates import match_coordinates_to_images
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr
import torchio as tio 

class TOMOPre2D(Dataset):
	num_classes = 1 
	default_resolution = [256, 256]

	def __init__(self, opt, split, size):
		super(TOMOPre2D, self).__init__()
		if split == 'train':
			self.data_dir = os.path.join(opt.data_dir, opt.train_img_txt)
		# self.coord_dir = os.path.join(opt.data_dir, opt.train_coord_txt)
			# self.data_dir = os.path.join(opt.data_dir, 'train_img_new_10304.txt')
			# self.data_dir = os.path.join(opt.data_dir,'10499_train_img_new.txt')
			# self.coord_dir = os.path.join(opt.data_dir, '10499_train_coord_newest_v2_less.txt')
			# self.coord_dir = os.path.join(opt.data_dir, '10304_train_coord_new_less.txt')
		# elif split == 'val':
		# 	self.data_dir = os.path.join(opt.data_dir, opt.val_img_txt)
		# 	self.coord_dir = os.path.join(opt.data_dir, opt.val_coord_txt)
		# 	# self.data_dir = os.path.join(opt.data_dir, '10304_new_test_img_all.txt')
		# 	# self.coord_dir = os.path.join(opt.data_dir, '10304_val_coord_new.txt')
		# 	# self.data_dir = os.path.join(opt.data_dir, '10499_test_img_new.txt')
		# 	# self.data_dir = os.path.join(opt.data_dir, '10499_test_img_new.txt')
		# 	# self.coord_dir = os.path.join(opt.data_dir, '10499_test_coord_newest_v2.txt')
		else:
			self.data_dir = os.path.join(opt.data_dir, opt.test_img_txt)
		# 	self.coord_dir = os.path.join(opt.data_dir, opt.test_coord_txt)
		# 	# self.data_dir = os.path.join(opt.data_dir, '10499_test_img_new.txt')
		# 	# self.coord_dir = os.path.join(opt.data_dir, '10499_test_coord_newest_v2.txt')
		# 	# self.data_dir = os.path.join(opt.data_dir, '10304_new_test_img_all.txt')
		# 	# self.coord_dir = os.path.join(opt.data_dir, '10304_val_coord_new.txt')
		# self.split = split 
		self.opt = opt 
		self.size = size
		# self.images, self.targets, self.names = self.load_data()
		self.tomos, self.names, self.subvols = self.load_data()
		# self.transforms = tio.Compose([
		# 	tio.RandomBlur(std=(0,1),p=0.15),
		# 	tio.RandomNoise(p=0.5),
		# 	tio.RandomAffine(scales=(1,1,1,1,1,1), translation=(0,0,0,0,0,0,),degrees=(0,60,0,0,0,0),p=0.75),
		# 	tio.transforms.Crop((self.size[0]//8, self.size[1]//8, self.size[2]//8)),
		# 	tio.transforms.ZNormalization(),
		# 	tio.transforms.RescaleIntensity(out_min_max=(-3,3)),
		# 	tio.transforms.ZNormalization()
		# 		]
		# 	)

		self.transforms = T.Compose([
			T.ToPILImage(),
			T.RandomHorizontalFlip(0.5),
			T.RandomVerticalFlip(0.5),
			T.CenterCrop(self.size[1]-self.size[1]//4),
			T.RandomRotation(50),
			T.ToTensor(),
			T.Normalize((0.5),(0.5))]
			)

		# self.scripted_transforms = torch.jit.script(transforms)
		# self.crop = tio.transforms.Crop((self.size[0]//4, self.size[1]//4, self.size[2]//4))
		self.num_samples = len(self.subvols)
		
		# self.width_z = width_z
		# self.width_xy = width_xy
		# print('hello tomo')
		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.num_samples

	def _downscale_coord(self,ann):
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
		
		image_list = pd.read_csv(self.data_dir, sep='\t')
		# coords = pd.read_csv(self.coord_dir, sep='\t')
		images = load_tomos_from_list(image_list.image_name, image_list.path, order='xzy', compress=True, denoise=False)
		# num_particles = len(coords)
		# ims, targets, names = self.match_images_targets(images, coords)
		# num_of_tomos = len(ims)
		# print('num_of_tomos', num_of_tomos)
		# print('coords', coords)
		tomos = []
		names = []
		sub_vols = []
		vol_size = np.asanyarray(self.size)
		for k, v in images.items():
			names.append(k)
			print('v', v.shape)
			v = v[40:80]
			tomos.append(v)
			blks = cutup(v, self.size ,(1,2,2))
			for i in range(blks.shape[0]):
				for j in range(blks.shape[1]):
					for k in range(blks.shape[2]):
						sub_vols.append(blks[i,j,k])
			del blks
			# blks = blks.reshape(-1, *vol_size)
			# for p in blks:
			# 	sub_vols.append(p)


		# hms = []
		# inds = []
		# gt_dets = []
		# for i in range(num_of_tomos):
		# 	tomo = ims[i]
		# 	coords = targets[i]
		# 	depth, height, width = tomo.shape[0], tomo.shape[1], tomo.shape[2]
		# 	num_objs = len(coords)
		# 	# print('num_objs', num_objs)
		# 	output_h, output_w = height // self.opt.down_ratio, width // self.opt.down_ratio
		# 	num_class = self.num_classes
		# 	# if self.opt.pn:
		# 	# 	hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
		# 	# else:
		# 	hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
		# 	ind = np.zeros((num_objs), dtype=np.int64)
		# 	draw_gaussian = draw_umich_gaussian_3d
		# 	gt_det = []
		# 	# all_coords = [np.arange(output_h), np.arange(output_w), np.arange(width)]
		# 	h = self.opt.bbox // self.opt.down_ratio
		# 	for k in range(num_objs):
		# 		ann = coords[k]
		# 		radius = gaussian_radius((math.ceil(h), math.ceil(h)))
		# 		radius = max(0, int(radius))
		# 		ann = self._downscale_coord(ann)
		# 		ann = np.array(ann)
		# 		ct_int = ann.astype(np.int32)
		# 		z_coord = ct_int[-1]
		# 		draw_gaussian(hm, ct_int, radius,  0, 0, 0, discrete=False)
		# 		ind[k] = ct_int[2] * (output_w * output_h) + ct_int[1] * output_w + ct_int[0]
		# 		# print('ann', ann)
		# 		gt_det.append(ann)
		# 	gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
		# # all_coords = np.array(all_coords)
		# 	tomos.append(tomo)
		# 	hms.append(hm)
		# 	inds.append(ind)
		# 	gt_dets.append(gt_det)
		# gt_dets = [gt_det]

		return tomos, names, sub_vols