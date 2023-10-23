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
from utils.pre_process import preprocess_label_constraints

class ParticleFewShotDataset(data.Dataset):
	
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

	# def _upscale_coord(self, ann):



	def __getitem__(self, index):
		if self.split == 'train':
			tomo = self.tomos[index]
			hm = self.hms[index]
			ind = self.inds[index]
			gt_det = self.gt_dets[index]
			lb_maps = self.lb_maps[index]
			labels = self.labels[index]
			print('index', index)
			print('names', self.names)
			print('tomo', tomo.shape)
			print('labels', labels.shape)
			# print('lb_maps', np.max(lb_maps))
			flip_prob = np.random.random()
			# print('gt_det', gt_det.shape)
			# print('gt_det x min', np.min(gt_det[:,0]))
			# print('gt_det y_min', np.min(gt_det[:,1]))
			name = self.names[index]
			depth, height, width = tomo.shape[0], tomo.shape[2], tomo.shape[1]
			num_of_samples = gt_det.shape[0]
			loc = random.randint(0, num_of_samples-1)
			# print('loc', loc)
			# print('height')
			# sampled_z_center = list(np.arange(10, depth-10))
			# sampled_x_center = list(np.arange(17, height//2 - 17))
			# # print('sampled_x_center', sampled_x_center)
			# sampled_y_center = list(np.arange(17, height//2 - 17))
			# x_c_r = random.sample(sampled_x_center, 1)[0]
			# y_c_r = random.sample(sampled_y_center, 1)[0]
			# z_c_r = random.sample(sampled_z_center, 1)[0]
			# offs_x_y = [-2, -1, 0, 1, 2]
			# 		# offs_z = [-2, -3, 3, 2]
			# offs_z = [-1, 0, 1]
			# off_x = random.sample(offs_x_y, 1)[0]
			# off_y = random.sample(offs_x_y, 1)[0]
			# off_z = random.sample(offs_z, 1)[0]
			x_c, y_c, z_c = gt_det[loc] 
			x_c, y_c, z_c = np.clip(x_c, 33, height//2 - 33), np.clip(y_c, 33, height//2 - 33), np.clip(z_c, 6, depth - 6)
			x_c, y_c, z_c = int(x_c), int(y_c), int(z_c)
			# print('x_c, y_c, z_c', x_c, y_c, z_c)
			up_xc, up_yc = int(x_c*self.opt.down_ratio), int(y_c*self.opt.down_ratio)
			# up_xc_r, up_yc_r = int(x_c_r * self.opt.down_ratio), int(y_c_r * self.opt.down_ratio)
			# sampled_z_center = list(np.arange(6, depth-6))
			# sampled_x_center = list(np.arange(33, height - 33))
			# sampled_y_center = list(np.arange(33, height - 33))
			# x_c = random.sample(negative_region_x, 1)[0]
			# y_c = random.sample(negative_region_y, 1)[0]
			# z_c = random.sample(negative_region_z, 1)[0]
			# cropped_tomo_c = tomo[z_c_r-10:z_c_r+10, up_yc_r-32:up_yc_r+32, up_xc_r-32:up_xc_r+32]
			# cropped_hm_c = hm[z_c_r-10:z_c_r+10, y_c_r-16:y_c_r+16, x_c_r-16:x_c_r+16]
			cropped_tomo = tomo[z_c-5:z_c+5, up_yc-64:up_yc+64, up_xc-64:up_xc+64]
			cropped_hm = hm[z_c-5:z_c+5, y_c-32:y_c+32, x_c-32:x_c+32]
			cropped_label = lb_maps[z_c-5:z_c+5, y_c-32:y_c+32, x_c-32:x_c+32]
			# print('cropped_hm', cropped_hm.shape)
			# print('cropped_tomo', cropped_tomo.shape)
			# print('cropped_label', cropped_label.sum())
			if flip_prob <= 0.5:
				cropped_tomo_aug = flip_lr(cropped_tomo)
				cropped_hm_aug = flip_lr(cropped_hm)
				cropped_label_aug = flip_lr(cropped_label)
			if flip_prob > 0.5:
				cropped_tomo_aug = flip_ud(cropped_tomo)
				cropped_hm_aug = flip_ud(cropped_hm)
				cropped_label_aug = flip_ud(cropped_label)
			ml_graph, cl_graph = preprocess_label_constraints(cropped_label)
			# print('ml_graph', ml_graph)

			# ret = {'input': img.astype(np.float32), 'input_aug': aug_img.astype(np.float32), 'hm': hm, 'hm_aug': hm_aug,'reg_mask': reg_mask, 'ind': ind, 'class': clas_z, 'gt_det': gt_det}
			ret ={'input':cropped_tomo.astype(np.float32),'input_aug': cropped_tomo.astype(np.float32), 'hm': cropped_hm, 'hm_aug': cropped_hm_aug, 'ind': ind, 'gt_det': gt_det, 'lb_map': cropped_label, 'ml_graph': ml_graph, 'cl_graph': cl_graph, 'flip_prob': flip_prob}
		else:
			tomo = self.tomos[index]
			hm = self.hms[index]
			ind = self.inds[index]
			gt_det = self.gt_dets[index]
			name = self.names[index]
			lb_maps = self.lb_maps[index]
			labels = self.labels[index]
			ret ={'input': tomo.astype(np.float32), 'hm': hm, 'ind': ind, 'gt_det': gt_det, 'lb_map':lb_maps}

		# if self.opt.contrastive:
		# 	ret.update({'soft_neg': soft_negs})



		# if self.opt.reg_offset:
		# 	ret.update({'reg':reg})
		if self.opt.debug > 0:
			
			meta = {'gt_det':gt_det, 'name': name}
			ret['meta'] = meta 
		return ret

		# tomo = self.images[index]
		# coords = self.targets[index]
		# name = self.names[index]
		# depth, height, width = img.shape[0], img.shape[1], img.shape[2]
		# num_objs = len(coords)

		# print('img', img.shape)
		# # print('num_objs', num_objs)
		# output_h, output_w = height // self.opt.down_ratio, width // self.opt.down_ratio
		# crop_w_o, crop_h_o = output_h // self.opt.patch_num, output_w // self.opt.patch_num
		# crop_w, crop_h = height // self.opt.patch_num, width //self.opt.down_ratio
		# num_class = self.num_classes
		# # negative_region_x = list(np.arange(20,output_h - 20))
		# # negative_region_y = list(np.arange(20,output_w - 20))
		# # negative_region_z = list(np.concatenate((np.arange(-10, -3), np.arange(3, 10))))
		# # negative_region_z = list(np.arange(0, 10))
		# # if self.split == 'train':
		# # 	hm = np.zeros((depth, output_h, output_w), dtype=np.float32) - 1
		# # else:
		# # 	hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
		# hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
		# hm_aug = np.zeros((depth, output_h, output_w), dtype=np.float32)
		# # hm[]
		# # hm[:10, :, :] = 0
		# # hm[-10:,:, :] = 0
		# # used_mask = np.zeros((depth, output_h, output_w), dtype=np.float32)
		# wh = np.zeros((num_objs, 1), dtype=np.float32)
		# reg = np.zeros((num_objs, 2), dtype=np.float32)
		# ind = np.zeros((num_objs), dtype=np.int64)
		# ind_aug = np.zeros((num_objs), dtype=np.int64)
		# clas_z = np.zeros((depth), dtype=np.float32)
		# reg_mask = np.zeros((num_objs), dtype=np.uint8)
		# # centers = np.zeros((num_objs, 3), dtype=np.float32)
		# draw_gaussian = draw_msra_gaussian_3d if self.opt.mse_loss else draw_umich_gaussian_3d

		# gt_det = []
		# gt_det_aug = []
		# if self.split == 'train':
		# 	flip_prob = np.random.random()
		# else:
		# 	flip_prob = 1.5

		# _flip_lr= False
		# _flip_ud = False
		# flip = False
		# if flip_prob < 0.5 
		# 	aug_img = flip_lr(img)
		# 	_flip_lr = True
		# 	flip = True 
		# else:
		# 	aug_img = flip_ud(img)
		# 	_flip_ud = True 
		# 	flip = True
		
		# h = self.opt.bbox // self.opt.down_ratio
		# for k in range(num_objs):
		# 	ann = coords[k]
		# 	# print('ann', ann)
		# 	if _flip_lr:
		# 		ann_aug = self._flip_coord_lr(ann, width)
		# 	if _flip_ud:
		# 		ann_aug = self._flip_coord_ud(ann, height)
		# 	radius = gaussian_radius((math.ceil(h), math.ceil(h)))
		# 	radius = max(0, int(radius))
		# 	# print('radius', radius)
		# 	radius = self.opt.hm_gauss if self.opt.mse_loss else radius
		# 	ann = self._downscale_coord(ann)
		# 	ann_aug = self._downscale_coord(ann_aug)
		# 	# print('ann', ann)
		# 	ann = np.array(ann)
		# 	ann_aug = np.array(ann_aug)
		# 	ct_int = ann.astype(np.int32)
		# 	ct_int_aug = ann_aug.astype(np.int32)
		# 	z_coord = ct_int[-1]
		# 	# centers[k] = ct_int
		# 	clas_z[int(z_coord)] = 1
		# 	draw_gaussian(hm, ct_int, radius)
		# 	draw_gaussian(hm_aug, ct_int, radius)
		# 	# wh[k] = h
		# 	# used_mask[z_coord, ct_int[1], ct_int[0]] = 1
		# 	# wrong expression
		# 	# ind[k] = (ct_int[1] * output_w + ct_int[0])*ct_int[2]
		# 	ind[k] = ct_int[2] * (output_w * output_h) + ct_int[1] * output_w + ct_int[0]
		# 	ind_aug[k] = ct_int_aug[2] * (output_w * output_h) + ct_int_aug[1] * output_w + ct_int_aug[0]
		# 	# print('ind', ct_int[2] * (output_w * output_h) + ct_int[1] * output_w + ct_int[0])
		# 	reg[k] = ann[:2] - ct_int[:2]
		# 	reg_mask[k] = 1
		# 	gt_det.append(ann)
		# 	gt_det_aug.append(ann_aug)
		# 	# if self.opt.contrastive:
		# 	# 	if self.split == 'train':
		# 	# 		offs_x_y = [-5,-4, -6, 4, 5, 6]
		# 	# 		offs_z = [-2, -3, 3, 2]
		# 	# 		off_x = random.sample(offs_x_y, 1)[0]
		# 	# 		off_y = random.sample(offs_x_y, 1)[0]
		# 	# 		off_z = random.sample(offs_z, 1)[0]
		# 	# 		# off_z = np.clip(off_z, 0, 127)
		# 	# 		off_coord = ann - np.array([off_x, off_y, off_z])
		# 	# 		# print('off_coord', off_coord)
		# 	# 		if off_coord[-1] >= 128:
		# 	# 			off_coord[-1] = 127
		# 	# 		off_coord = np.clip(off_coord, 0, 255)
		# 	# 		# print('off_coord', off_coord)
		# 	# 		used_mask[off_coord[-1], off_coord[-2], off_coord[-3]] = 1
		# 	# 		soft_negs.append(off_coord)
		# 	# 	else:
		# 	# 		soft_negs.append([0,0,0])
		# 	# 		used_mask[0,0,0] = 1


		# gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
		# gt_det_aug = np.array(gt_det, dtype=np.float32) if len(gt_det_aug) > 0 else np.zeros((1,3), dtype=np.float32)
		# if self.opt.contrastive:
		# 	# if self.split == 'train:'
		# 	print('generating hard negative examples')
		# 	# hard_negs = []
		# 	num_of_hard_negs = 30
		# 	for i in range(num_of_hard_negs):
		# 		x_neg = random.sample(negative_region_x, 1)[0]
		# 		y_neg = random.sample(negative_region_y, 1)[0]
		# 		z_neg = random.sample(negative_region_z, 1)[0]
		# 		neg_ann = np.array([x_neg, y_neg, z_neg])
		# 		hm[z_neg, y_neg, x_neg] = 0
		# 		used_mask[z_neg, y_neg, x_neg] = 1
		# 		soft_negs.append(neg_ann)
		# 	# hard_negs = np.array(hard_negs, dtype=np.float32)
		# 	soft_negs = np.array(soft_negs, dtype=np.float32)

			
		






