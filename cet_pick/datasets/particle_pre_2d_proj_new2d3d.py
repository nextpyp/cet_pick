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

class ParticlePreProjDataset2D3D(data.Dataset):
    
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
    def convert_tomo_to_tilt(self, tomo_coord, angle, tomo_size = [512, 512, 256]):
        angle = angle*np.pi/180
        tomo_x, tomo_y, tomo_z = tomo_coord[0], tomo_coord[1], 256 - tomo_coord[2]
        tilt_x  = (tomo_x - tomo_size[0]//2) * math.cos(angle) + (tomo_z - tomo_size[-1]//2) * math.sin(angle) + tomo_size[0]//2
        tilt_y = tomo_y
        return int(tilt_x), int(tilt_y)

    def extract_patches(self, v, tomo_coord, angles, tomo_size=[512, 512, 256]):
        patches = None 
        for ind, an in enumerate(angles):
            tx, ty = self.convert_tomo_to_tilt(tomo_coord, an, tomo_size)
            if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[2]//2 or ty >= self.tomo_size[2]-self.size[1]//2:
                        continue 
            patch = v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2]
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

    def _convert_coord1d_to_3d(self, coord, width, height):

        z = coord // (width * height)
        x = (coord - z * width * height) % width 
        y = (coord - z * width * height - x) // width

        return [x,y,z]

    def __getitem__(self, index):
        # if self.split == 'train':
            
        # tomo = self.tomos[i]
        sub_vol_set = self.subvols_sets[index]
        sub_vol_set_3d = self.sub_vols_sets_3d[index]
        sub_vol = sub_vol_set[0]
        sub_vol3d = sub_vol_set_3d[0]
        sub_vols = torch.cat([sub_vol, sub_vol3d], dim=0)
        num_of_augs = len(sub_vol_set)
        curr_coord = self.coords[index]
        use_aug = np.random.randint(1, num_of_augs)
        aug_patch = sub_vol_set[use_aug]
        aug_patch3d = sub_vol_set_3d[use_aug]
        aug_patches = torch.cat([aug_patch, aug_patch3d], dim=0)
        # print('aug_patches', aug_patches.shape)
        aug_2 = self.weak_transforms(aug_patches)
       
        aug_1 = self.transforms(sub_vols)
       
        ret = {'input': aug_1[0:1].float(), 'input_3d': aug_1[1:2].float(), 'input_aug': aug_2[0:1].float(), 'input_aug_3d': aug_2[1:2].float(), 'coord':curr_coord}
        return ret

       