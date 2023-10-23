from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import mrcfile
import cv2
import json
import os
import pandas as pd 

import torch.utils.data as data
from cet_pick.utils.loader import load_tomos_from_list
from cet_pick.utils.coordinates import match_coordinates_class_to_images


class TOMO_Multi(Dataset):
	num_classes = 16 
	default_resolution = [512, 512]

	def __init__(self, opt, split):
		super(TOMO_Multi, self).__init__()
		if split == 'train':
			self.data_dir = os.path.join(opt.data_dir, 'train_miss_img.txt')
			self.coord_dir = os.path.join(opt.data_dir, 'train_coord_partial.txt')
		elif split == 'val':
			self.data_dir = os.path.join(opt.data_dir, 'val_miss_img.txt')
			self.coord_dir = os.path.join(opt.data_dir, 'val_coord.txt')
		else:
			self.data_dir = os.path.join(opt.data_dir, 'test_miss_img.txt')
			self.coord_dir = os.path.join(opt.data_dir, 'test_coord.txt')
		self.split = split 
		self.opt = opt 

		self.images, self.targets, self.names = self.load_data()
		self.num_samples = len(self.images)

		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.num_samples

	def match_images_targets(self, images, targets):
		matched = match_coordinates_class_to_images(targets, images)
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

		# print('ims', len(ims))
		# print(ims[0].shape)
		# print('targets', len(targets))
		# print('targets', targets[0])
		# print(targets[0])
		# print(names)

		return ims, targets, names












