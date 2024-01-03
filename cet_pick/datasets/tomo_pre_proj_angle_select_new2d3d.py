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
import random
from cet_pick.utils.loader import load_tomos_from_list, cutup, load_tomos_and_angles_from_list, load_tomo_all_and_angles_from_list
from cet_pick.utils.coordinates import match_coordinates_to_images
from cet_pick.utils.image import get_potential_coords, get_potential_coords_pyramid
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr, CornerErasing, CenterOut,AdjustBrightness,InvertColor, FixedRotation
import torchio as tio 
from multiprocessing import Pool


class TOMOPreProjAngleSelect2D3D(Dataset):
    num_classes = 1 
    default_resolution = [256, 256]

    def __init__(self, opt, split, size, low = -20, up = 20, sigma1=2, K=5000):
        super(TOMOPreProjAngleSelect2D3D, self).__init__()
        if split == 'train':
            self.data_dir = os.path.join(opt.data_dir, opt.train_img_txt)
        else:
            self.data_dir = os.path.join(opt.data_dir, opt.test_img_txt)

        self.opt = opt 
        self.size = size
        self.crop_size_x, self.crop_size_y = int(np.ceil(self.size[1]*np.sqrt(1))), int(np.ceil(self.size[2]*np.sqrt(1)))
        self.coords = []
        self.names_all = []
        self.low = low 
        self.up = up 
        self.split = split
        self.hm_shape = None
        self.sigma1 = sigma1 
        self.K = K
        self.tomos, self.names, self.subvols, self.subvols_sets, self.sub_vols_3d, self.sub_vols_sets_3d = self.load_data()

        self.transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            # T.GaussianBlur(3, sigma=(0.1,0.2)),
            T.RandomRotation(30),
            # T.RandomAffine(degrees=45, translate=(0.1,0.2)),
            # AdjustBrightness(),
            # InvertColor(),
            T.CenterCrop(self.size[1]),
            # T.ColorJitter(0.5, 0.2, 0.3, 0.1),
            T.ToTensor(),
            CornerErasing(p=0.5,scale = (0.01, 0.02), ratio = (0.5, 1.5)),
            FixedRotation(),
            # CenterOut(crop_dim = 12),
            T.Normalize((self.mean_subvols, self.mean_subvols3d),(self.std_subvols, self.std_subvols3d))]
            )

        self.weak_transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            # T.GaussianBlur(3, sigma=(0.05,0.15)),
            # T.RandomAdjustSharpness(3, p=0.5),
            # AdjustBrightness(p=0.5, brightness_factor=1.2),
            # T.RandomRotation(360),
            # T.RandomAffine(degrees=45, translate=(0.1,0.2)),
            T.CenterCrop(self.size[1]),
            T.ToTensor(),
            CornerErasing(p=0.5,scale = (0.01, 0.02), ratio = (0.5, 1.5)),
            FixedRotation(),
            # CenterOut(crop_dim = 24),
            T.Normalize((self.mean_subvols, self.mean_subvols3d),(self.std_subvols, self.std_subvols3d))]
            )

        self.num_samples = len(self.subvols)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def convert_tomo_to_tilt(self, tomo_coord, angle, tomo_size = [512, 512, 256]):
        angle = angle*np.pi/180
        tomo_x, tomo_y, tomo_z = tomo_coord[0], tomo_coord[1], tomo_size[-1] - tomo_coord[2]
        tilt_x  = (tomo_x - tomo_size[0]//2) * math.cos(angle) + (tomo_z - tomo_size[-1]//2) * math.sin(angle) + tomo_size[0]//2
        tilt_y = tomo_y
        return int(tilt_x), int(tilt_y)
        
    def _downscale_coord(self,ann):
        x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]
        return [x,y,z]

    def extract_3d_tomo(self, rec, tomo_coord):
        x, y, z = tomo_coord[0], tomo_coord[1], tomo_coord[2]
        # print('tomo_coord', tomo_coord)
        if self.opt.compress:
            z = int(z // 2)
        patch_3d = rec[z, y - self.crop_size_y//2:y + self.crop_size_y//2, x - self.crop_size_x//2:x + self.crop_size_x//2].copy()
        patch_3d = (patch_3d - np.min(patch_3d))/(np.max(patch_3d) - np.min(patch_3d))

        patch_3d = torch.tensor(patch_3d)
        patch_3d = patch_3d.unsqueeze(0).float()
        return patch_3d

    def extract_patches(self, v, tomo_coord, angles, tomo_size=[512, 512, 256]):
        patches = None 
        for ind, an in enumerate(angles):
            tx, ty = self.convert_tomo_to_tilt(tomo_coord, an, tomo_size)
            if tx <= self.crop_size_x//1.8 or tx >= self.tomo_size[1]-self.crop_size_x//1.8 or ty <= self.crop_size_y//1.8 or ty >= self.tomo_size[1]-self.crop_size_y//1.8:
                        continue 
            patch = v[ind, ty - self.crop_size_y//2: ty + self.crop_size_y//2, tx - self.crop_size_x//2: tx + self.crop_size_x//2].copy()
            if patches is None:
                patches = patch 
            else:
                patches += patch 
        if patches is not None:
            if np.min(patches) != np.max(patches):
                patches = (patches - np.min(patches))/(np.max(patches) - np.min(patches))
                patches = torch.tensor(patches)
                patches = patches.unsqueeze(0).float()
                return patches
            else:
                return None
        else:
            return None

    # @staticmethod
    def gaussian_1d(self, shape, sigma = 2):
        m = (shape-1.)/2. 
        z = np.ogrid[-m:m+1]
        h = np.exp(-(z*z)/(2*sigma*sigma)) 
        return h

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
        tilt_images, recs, angles = load_tomo_all_and_angles_from_list(image_list.image_name, image_list.tilt_path, image_list.rec_path, image_list.angle_path, compress=self.opt.compress, denoise=self.opt.gauss)
        tomos = {}
        names = []
        sub_vols = []
        sub_vols_3d = []
        sub_vols_sets = []
        sub_vols_sets_3d = []
        self.angles = {}
        vol_size = np.asanyarray(self.size)
        save_npy_rec = False
        save_npy_v = False
        for k, v in tilt_images.items():
            names.append(k)
            angle = angles[k]
            rec = recs[k]
            tomo_size_z, tomo_size_y, tomo_size_x = rec.shape
            self.tomo_size = [tomo_size_x, tomo_size_y, tomo_size_z]
            if self.opt.compress:
                tomo_size_z = tomo_size_z * 2
            scores, positions = get_potential_coords_pyramid(rec, sigmas = self.sigma1)
            num_of_coords = positions.shape[0]
            # print('angle', angle)
            used_tilts = np.where((angle >= self.low) & (angle <= self.up))[0]
            used_angles = angle[np.where((angle >= self.low) & (angle <= self.up))]
            used_v = v[used_tilts]
            self.angles[k] = used_angles
            for p in range(num_of_coords):
                patches = None 
                curr_coord = positions[p]

                tomo_x, tomo_y, tomo_z = positions[p][0], positions[p][1], positions[p][2]
                if self.opt.compress:
                    tomo_z = tomo_z * 2
                # check tomo coords:
                if tomo_x > self.crop_size_x//1.8 and tomo_x < self.tomo_size[1]-self.crop_size_x//1.8 and tomo_y >= self.crop_size_y//1.8 and tomo_y <= self.tomo_size[1]-self.crop_size_y//1.8:

                    x_c_1, y_c_1, z_c_1 = tomo_x, tomo_y, tomo_z + 1
                    x_c_2, y_c_2, z_c_2 = tomo_x, tomo_y, tomo_z - 1
                    x_c_3, y_c_3, z_c_3 = tomo_x-1, tomo_y, tomo_z - 1
                    x_c_4, y_c_4, z_c_4 = tomo_x, tomo_y+1, tomo_z - 1
                    aug_all = [[x_c_1, y_c_1, z_c_1],[x_c_2, y_c_2, z_c_2],[x_c_3, y_c_3, z_c_3],[x_c_4, y_c_4, z_c_4]]
                    patch_orig = self.extract_patches(used_v, [tomo_x, tomo_y, tomo_z], used_angles, tomo_size = [tomo_size_x, tomo_size_y, tomo_size_z])
                    if not patch_orig is None:
                        patch_orig_3d = self.extract_3d_tomo(rec, [tomo_x, tomo_y, tomo_z])
                        if self.split == 'train':
                            patch_sets = [patch_orig]
                            patch3d_sets = [patch_orig_3d]
                            for aug_ind in aug_all:
                                patch_aug = self.extract_patches(used_v, aug_ind, used_angles,tomo_size = [tomo_size_x, tomo_size_y, tomo_size_z])
                                if patch_aug is None:
                                    continue 
                                patch_aug_3d = self.extract_3d_tomo(rec, aug_ind)
                                patch_sets.append(patch_aug)
                                patch3d_sets.append(patch_aug_3d)

                            if len(patch_sets) > 1:
                                sub_vols_3d.append(patch_orig_3d)
                                sub_vols_sets_3d.append(patch3d_sets)
                                sub_vols_sets.append(patch_sets)
                                sub_vols.append(patch_orig)
                                self.coords.append(positions[p])
                                self.names_all.append(k)
                        else:
                            sub_vols.append(patch_orig)
                            sub_vols_3d.append(patch_orig_3d)
                            self.coords.append(positions[p])
                            self.names_all.append(k)


               

            tomos[k] = used_v
        self.mean_subvols3d = torch.mean(torch.stack(sub_vols_3d))
        self.std_subvols3d = torch.std(torch.stack(sub_vols_3d))
        self.mean_subvols = torch.mean(torch.stack(sub_vols))
        self.std_subvols = torch.std(torch.stack(sub_vols))
            

        return tomos, names, sub_vols, sub_vols_sets, sub_vols_3d, sub_vols_sets_3d