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
import random
import torch.utils.data as data
import torch
from cet_pick.utils.loader import load_tomos_from_list, load_tomos_and_angles_from_list
from cet_pick.utils.coordinates import match_coordinates_to_images, match_coordinates_class_to_images
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr

class TOMOPreTestAngle(Dataset):
	num_classes = 1 
	default_resolution = [256, 256]

	def __init__(self, opt, split,size, low = -20, up = 20, tomo_size = (512, 512, 256), slice_range = (128, 150)):
		super(TOMOPreTestAngle, self).__init__()
		
		self.data_dir = os.path.join(opt.data_dir, opt.test_img_txt)
		self.coord_dir = os.path.join(opt.data_dir, opt.test_coord_txt)
		self.size = size
		self.split = split 
		self.opt = opt 
		# print('with_gold', with_gold)
		# if with_gold:
		# 	self.gold_dir = '/nfs/bartesaghilab2/qh36/joint_data/EMPIAR_10304/gold/tilt6_gold_train_coords.txt'
		# 	# self.gold_dir = '/nfs/bartesaghilab2/qh36/joint_data/EMPIAR_10499/unsup_train/TS_02_gold3d_coord.txt'
		# self.with_gold = with_gold
		# self.images, self.targets, self.names = self.load_data()
		self.low = low 
		self.up = up 
		self.tomo_size = tomo_size
		self.slice_range = slice_range
		self.tomos, self.hms, self.gt_dets, self.names, self.subvols, self.labels = self.load_data()
		print('subvol',len(self.subvols))
		print('coord',len(self.gt_dets))
		# self.num_samples = len(self.tomos)
		self.num_samples = len(self.subvols)
		

		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.num_samples

	def _downscale_coord(self,ann):
		x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]
		return [x,y,z]

	def match_images_targets(self, images, targets, with_class=False):
		if with_class:
			matched = match_coordinates_class_to_images(targets, images)
		else:
			matched = match_coordinates_to_images(targets, images)
		tomos = []
		targets = []
		names = []
		for key in matched:
			names.append(key)
			tomos.append(matched[key]['tomo'])
			targets.append(matched[key]['coord'])
		return tomos, targets, names

	def convert_tomo_to_tilt(self, tomo_coord, angle, tomo_size = [512, 512, 256]):
	    angle = angle*np.pi/180
	    tomo_x, tomo_y, tomo_z = tomo_coord[0], tomo_coord[1], tomo_coord[2]
	    tilt_x  = (tomo_x - tomo_size[0]//2) * math.cos(angle) + (tomo_z - tomo_size[-1]//2) * math.sin(angle) + tomo_size[0]//2
	    tilt_y = tomo_y
	    return int(tilt_x), int(tilt_y)


	def load_data(self, image_ext=''):
		
		image_list = pd.read_csv(self.data_dir, sep='\t')
		coords = pd.read_csv(self.coord_dir, sep='\t')
		# if self.with_gold:
		# 	gold_coords = pd.read_csv(self.gold_dir, sep='\t')
		# images = load_tomos_from_list(image_list.image_name, image_list.path,order='zxy', compress=False, denoise=True, tilt=True)
		images, angles = load_tomos_and_angles_from_list(image_list.image_name, image_list.path, image_list.angle_path, order='zxy', compress=False, denoise=self.opt.gauss, tilt=True)
		num_particles = len(coords)
		ims, targets, names = self.match_images_targets(images, coords, with_class=True)
		# if self.with_gold:
		# 	_, gold_targets,_ = self.match_images_targets(images, gold_coords)
		num_of_tomos = len(ims)	
		print('num_of_tomos,', num_of_tomos)
		print('targets', targets)
		print('names', names)
		# print('gold_targets', gold_targets)
		dd, hh, ww = int(self.size[0]//2), int(self.size[1]//2), int(self.size[2]//2)
		# print(dd,hh,ww)
		tomos = []
		hms = []
		subvols = []
		lbs = []
		# inds = []
		gt_dets = []
		used_coord = []
		for i in range(num_of_tomos):
			tomo = ims[i]
			coords = targets[i]
			name = names[i]
			angle = angles[name]
			# print('angle', angle)
			used_tilts = np.where((angle >= self.low) & (angle <= self.up))[0]
			used_angles = angle[np.where((angle >= self.low) & (angle <= self.up))]
			used_v = tomo[used_tilts]
			# if self.with_gold:
			# 	gold_coords = gold_targets[i]
			depth, height, width = tomo.shape[0], tomo.shape[1], tomo.shape[2]
			num_objs = len(coords)
			
			# print('tomo', tomo.shape)
			# print('num_objs', num_objs)
			output_h, output_w = height, width
			# num_class = self.num_classes
			# if self.opt.pn:
			# 	hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
			# else:
			hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
			# ind = np.zeros((num_objs), dtype=np.int64)
			draw_gaussian = draw_umich_gaussian_3d
			gt_det = []
			
			# all_coords = [np.arange(output_h), np.arange(output_w), np.arange(width)]
			h = self.opt.bbox // self.opt.down_ratio
			for k in range(num_objs):
				ann = coords[k][:3]
				lb = coords[k][-1]
				# print('lb', lb)
				# if lb == 15:
				# 	print(ann)
				# 	print(lb)
				radius = gaussian_radius((math.ceil(h), math.ceil(h)))
				radius = max(0, int(radius))
				ann = self._downscale_coord(ann)
				ann = np.array(ann)
				x, y, z = int(ann[0]), int(ann[1]), int(ann[2])
				# print('xyz', x,y,z)
				# if lb == 1 or lb == 2 or lb ==3:
				patches = None 
				for ind,an in enumerate(used_angles):
					tx, ty = self.convert_tomo_to_tilt(ann, an, self.tomo_size)
					if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[2]//2 or ty >= self.tomo_size[2]-self.size[1]//2:
						continue 
					patch = used_v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2]
					# print('patch', patch.shape)
					if patches is None:
						patches = patch 
					else:
						# print('wtf patch', patches.shape)
						# print('wtf patch2,',patch.shape)
						patches += patch 
					
				if patches is not None:
					# print('patches', patches.shape)
					patches -= patches.min()
					if patches.max() != 0:
						patches /= patches.max()
						patches = torch.from_numpy(patches)
						patches = patches.unsqueeze(0).float()
						# print('patches', patches.shape)
						subvols.append(patches)
						used_coord.append(ann)



				# if 1:
				# 	if z > dd+2 and z < depth-dd-2 and y > hh+10 and y < height-hh-10 and x > ww+10 and x < width-ww-10:
				# 		# print(cr)
				# 		cropped_subvol = tomo[z-dd:z+dd+1, y-hh:y+hh, x-ww:x+ww]
				# 		# if cropped_subvol.shape[1] != 48 or cropped_subvol.shape[2] != 48:
				# 		# 	print(cropped_subvol.shape)
				# 		# 	print('index', x,y,z)
				# 		subvols.append(cropped_subvol)
						# print('cropped pos subvols', cropped_subvol.shape)
						lbs.append(lb)
				ct_int = ann.astype(np.int32)
					# z_coord = ct_int[-1]
				draw_gaussian(hm, ct_int, radius,  0, 0, 0, discrete=False)
					# ind[k] = ct_int[2] * (output_w * output_h) + ct_int[1] * output_w + ct_int[0]
					# print('ann', ann)
				gt_det.append(ann)
			# gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
			# if self.with_gold:
			# 	print('with gold')
			# 	num_golds = len(gold_coords)
			# 	for j in range(num_golds):
			# 		ann = gold_coords[j]
			# 		print(ann)
			# 		radius = gaussian_radius((math.ceil(h), math.ceil(h)))
			# 		radius = max(0, int(radius))
			# 		ann = self._downscale_coord(ann)
			# 		ann = np.array(ann)
			# 		x, y, z = int(ann[0]), int(ann[1]), int(ann[2])
			# 		# print('xyz', x,y,z)
			# 		if z > dd+2 and z < depth-dd-2 and y > hh+2 and y < height-hh-2 and x > ww+2 and x < width-ww-2:
			# 			# print(cr)
			# 			cropped_subvol = tomo[z-dd:z+dd, y-hh:y+hh, x-ww:x+ww]
			# 			subvols.append(cropped_subvol)
			# 			lbs.append(2)
			# 		ct_int = ann.astype(np.int32)
			# 			# z_coord = ct_int[-1]
			# 		draw_gaussian(hm, ct_int, radius,  0, 0, 0, discrete=False)
			# 			# ind[k] = ct_int[2] * (output_w * output_h) + ct_int[1] * output_w + ct_int[0]
			# 			# print('ann', ann)
			# 		gt_det.append(ann)
			# gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
			negatives = np.where(hm == 0) 
			num_negs = negatives[0].shape[0]
			sample = random.sample(list(np.arange(num_negs)), num_objs)
			for j in range(num_objs):
				patches = None 
				z_n, y_n, x_n = np.clip(negatives[0][sample[j]], dd+10, depth-dd-10), np.clip(negatives[1][sample[j]], hh+10, height-hh-10), np.clip(negatives[2][sample[j]], ww+10, width-ww-10)
				for ind,an in enumerate(used_angles):
					tx, ty = self.convert_tomo_to_tilt([x_n, y_n, z_n], an, self.tomo_size)
					if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[2]//2 or ty >= self.tomo_size[2]-self.size[1]//2:
						continue 
					patch = used_v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2]
					if patches is None:
						patches = patch 
					else:
						# print('wtf patch', patches.shape)
						# print('wtf patch2,',patch.shape)
						patches += patch 
				if not patches is None:
					patches -= patches.min()
					if patches.max() != 0:
						patches /= patches.max()
						patches = torch.tensor(patches)
						patches = patches.unsqueeze(0).float()
						# print('patches', patches.shape)
						subvols.append(patches)
			# 	cropped_subvol = tomo[z_n-dd:z_n+dd, y_n-hh:y_n+hh, x_n-ww:x_n+ww]
			# 	# print('cropped_subvol neg', cropped_subvol.shape)
			# 	subvols.append(cropped_subvol)
						lbs.append(2)
						used_coord.append(np.array([x_n, y_n, z_n]))

			# print('lb length', len(subvols))
		# all_coords = np.array(all_coords)
			tomos.append(tomo)
			hms.append(hm)
			# inds.append(ind)
			gt_dets.append(gt_det)
		# gt_dets = [gt_det]

		return tomos, hms, used_coord, names, subvols, lbs