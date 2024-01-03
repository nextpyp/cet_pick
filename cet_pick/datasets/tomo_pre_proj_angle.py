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
from cet_pick.utils.loader import load_tomos_from_list, cutup, load_tomos_and_angles_from_list
from cet_pick.utils.coordinates import match_coordinates_to_images
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr, CornerErasing, CenterOut
import torchio as tio 
from multiprocessing import Pool

class TOMOPreProjAngle(Dataset):
	num_classes = 1 
	default_resolution = [256, 256]

	def __init__(self, opt, split, size, low = -20, up = 20, tomo_size = (512, 512, 256), slice_range = (128, 150)):
		super(TOMOPreProjAngle, self).__init__()
		if split == 'train':
			self.data_dir = os.path.join(opt.data_dir, opt.train_img_txt)
		else:
			self.data_dir = os.path.join(opt.data_dir, opt.test_img_txt)
		self.opt = opt 
		self.size = size
		self.coords = []
		self.low = low 
		self.up = up 
		self.tomo_size = tomo_size
		self.split = split
		self.hm_shape = None
		# self.images, self.targets, self.names = self.load_data()
		self.slice_range = slice_range
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
			T.RandomRotation(60),
			# T.RandomAffine(degrees=30, translate=(0.1,0.2)),
			T.CenterCrop(self.size[1]-self.size[1]//4),
			T.ToTensor(),
			# CornerErasing(p=0.5,scale = (0.01, 0.015), ratio = (0.5, 1.5)),
			CenterOut(crop_dim = 24),
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

	def convert_tomo_to_tilt(self, tomo_coord, angle, tomo_size = [512, 512, 256]):
	    angle = angle*np.pi/180
	    tomo_x, tomo_y, tomo_z = tomo_coord[0], tomo_coord[1], 256 - tomo_coord[2]
	    tilt_x  = (tomo_x - tomo_size[0]//2) * math.cos(angle) + (tomo_z - tomo_size[-1]//2) * math.sin(angle) + tomo_size[0]//2
	    tilt_y = tomo_y
	    return int(tilt_x), int(tilt_y)
	def _downscale_coord(self,ann):
		x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]
		return [x,y,z]

	def extract_patches(self, v, tomo_coord, angles, tomo_size=[512, 512, 256]):
		patches = None 
		for ind, an in enumerate(angles):
			tx, ty = self.convert_tomo_to_tilt(tomo_coord, an, tomo_size)
			if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[2]//2 or ty >= self.tomo_size[2]-self.size[1]//2:
						continue 
			patch = used_v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2]
			if patches is None:
				patches = patch 
			else:
				patches += patch 
		if patches is not None:
			patches -= patches.min()
			if patches.max() != 0:
				patches /= patches.max()
				patches = torch.tensor(patches)
				patches = patches.unsqueeze(0).float()
				return patches
		else:
			return None


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
		# images = load_tomos_from_list(image_list.image_name, image_list.path, order='zxy', compress=False, denoise=False, tilt=True)
		images, angles = load_tomos_and_angles_from_list(image_list.image_name, image_list.rec_path, image_list.angle_path, order='zxy', compress=False, denoise=self.opt.gauss, tilt=True)
		# print('self.opt.gauss', self.opt.gauss)
		if self.split == 'train':
			x_s, y_s, z_s = np.arange(self.size[1]//2, self.tomo_size[0]-self.size[1]//2, 2), np.arange(self.size[1]//2, self.tomo_size[0]-self.size[1]//2, 2), np.arange(50, 180, 2)
			xv, yv, zv = np.meshgrid(x_s, y_s, z_s, indexing = 'xy')
			positions = np.vstack([xv.ravel(), yv.ravel(),zv.ravel()])
			num_of_coords = positions.shape[-1]
		elif self.split == 'test':
			x_s, y_s, z_s = np.arange(self.size[1]//2, self.tomo_size[0]-self.size[1]//2, 1), np.arange(self.size[1]//2, self.tomo_size[0]-self.size[1]//2, 1), np.arange(self.slice_range[0], self.slice_range[1], 2)
			xv, yv, zv = np.meshgrid(x_s, y_s, z_s, indexing = 'xy')
			positions = np.vstack([xv.ravel(), yv.ravel(),zv.ravel()])
			num_of_coords = positions.shape[-1]
		else:
			ims, targets, names = self.match_images_targets(images, coords)
		tomos = []
		names = []
		sub_vols = []
		vol_size = np.asanyarray(self.size)
		for k, v in images.items():
			names.append(k)
			angle = angles[k]
			used_tilts = np.where((angle >= self.low) & (angle <= self.up))[0]
			used_angles = angle[np.where((angle >= self.low) & (angle <= self.up))]

			used_v = v[used_tilts]
			

			for p in range(num_of_coords):
				patches = None 
				for ind,an in enumerate(used_angles):
					tx, ty = self.convert_tomo_to_tilt(positions[:,p], an, self.tomo_size)

					if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[2]//2 or ty >= self.tomo_size[1]-self.size[1]//2:
						continue 

					patch = used_v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2]
					if patches is None:
						patches = patch 
					else:
						patches += patch 
				if not patches is None:
					patches -= patches.min()
					if patches.max() != 0:
						patches /= patches.max()
						patches = torch.tensor(patches)
						patches = patches.unsqueeze(0).float()
						sub_vols.append(patches)
						self.coords.append(positions[:,p])

			tomos.append(used_v)


		return tomos, names, sub_vols