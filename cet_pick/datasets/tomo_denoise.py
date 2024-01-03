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
import torchvision.transforms as T
import math 
from PIL import Image
import torch.utils.data as data
from cet_pick.utils.loader import load_tomos_from_list_nopre
from cet_pick.utils.coordinates import match_coordinates_to_images
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr,RandomCropNoBorder

class TOMODenoise(Dataset):
	num_classes = 1 
	default_resolution = [256, 256]

	def __init__(self, opt, split):
		super(TOMODenoise, self).__init__()
		if split == 'train':
			self.data_dir = os.path.join(opt.data_dir, opt.train_img_txt)
			# self.coord_dir = os.path.join(opt.data_dir, opt.train_coord_txt)
		
		elif split == 'val':
			self.data_dir = os.path.join(opt.data_dir, opt.val_img_txt)
			# self.coord_dir = os.path.join(opt.data_dir, opt.val_coord_txt)

		else:
			self.data_dir = os.path.join(opt.data_dir, opt.test_img_txt)
			# self.coord_dir = os.path.join(opt.data_dir, opt.test_coord_txt)

		self.split = split 
		self.opt = opt 


		# if self.split == 'train' or self.split == 'val':
			# self.tomos, self.hms, self.inds, self.gt_dets, self.names, self.all_anns = self.load_data()
		self.single_tomos, self.tomo_list_inds, self.tomo_names = self.load_data()
			# self.num_samples = len(self.tomos)
			# if self.split == 'train':
		self.num_samples = len(self.single_tomos)

		self.to_tensor = T.ToTensor()
		self.transforms = T.Compose([
			RandomCropNoBorder(128, exclude=200),
			T.ToTensor()])
		# 	else:
		# 		self.num_samples = len(self.names)
		# else:
		# 	self.names, self.paths, _ = self.load_data()
		# 	self.num_samples = len(self.names)
		

		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.num_samples


	

	def load_data(self, image_ext=''):
		
		# if self.split == 'train' or self.split == 'val':
		image_list = pd.read_csv(self.data_dir, sep='\t')
		images = load_tomos_from_list_nopre(image_list.image_name, image_list.path, order=self.opt.order, compress=self.opt.compress, denoise=self.opt.gauss, tilt=True)
		num_of_tomos = len(images)

		all_tomo_names = []
		all_tomos = []
		all_tomo_inds = []

	
		for k, v in images.items():
			num_of_frames = v.shape[0]
			# print('v shape', v.shape)
			# all_tomo_ind.append(v)
			# curr_tomo = []
			for j in range(num_of_frames):
				curr_tlt = v[j]
				curr_tlt = cv2.normalize(curr_tlt, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_32F)
				curr_tlt = (curr_tlt*255).astype(np.uint8)
				curr_tlt_f = Image.fromarray(curr_tlt, 'L')
				
				all_tomos.append(curr_tlt_f)
				all_tomo_names.append(k)
				all_tomo_inds.append(j)
				# curr_tomo.append(curr_tlt_f)
			# all_tomo_ind.append(curr_tomo)



		return all_tomos, all_tomo_inds, all_tomo_names
		

		