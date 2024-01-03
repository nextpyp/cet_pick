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
from cet_pick.utils.loader import load_tomos_from_list, cutup, load_tomos_and_angles_from_list, load_tomo_all_and_angles_from_list
from cet_pick.utils.coordinates import match_coordinates_to_images
from cet_pick.utils.image import get_potential_coords
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr, CornerErasing, CenterOut
import torchio as tio 
from multiprocessing import Pool


class TOMOSCAN2D3DProjAngleSelect(Dataset):
    num_classes = 1 
    default_resolution = [256, 256]

    def __init__(self, opt, split, num_neighbors=None, names = None, name = None):
        super(TOMOSCAN2D3DProjAngleSelect, self).__init__()
        if split == 'train':
            self.data_dir = os.path.join(opt.simsiam_dir, opt.train_img_txt)
            self.ind_dir = os.path.join(opt.simsiam_dir, opt.train_coord_txt)
            self.names_dir = names 
            self.name = name 
            # self.name_dir = '/nfs/bartesaghilab/qh36/3D_picking/cet_pick_github/cet_pick/cet_pick/'
            print('data_dir', self.data_dir)
        # self.coord_dir = os.path.join(opt.data_dir, opt.train_coord_txt)
            # self.data_dir = os.path.join(opt.data_dir, 'train_img_new_10304.txt')
            # self.data_dir = os.path.join(opt.data_dir,'10499_train_img_new.txt')
            # self.coord_dir = os.path.join(opt.data_dir, '10499_train_coord_newest_v2_less.txt')
            # self.coord_dir = os.path.join(opt.data_dir, '10304_train_coord_new_less.txt')
        # elif split == 'val':
        #   self.data_dir = os.path.join(opt.data_dir, opt.val_img_txt)
        #   self.coord_dir = os.path.join(opt.data_dir, opt.val_coord_txt)
        #   # self.data_dir = os.path.join(opt.data_dir, '10304_new_test_img_all.txt')
        #   # self.coord_dir = os.path.join(opt.data_dir, '10304_val_coord_new.txt')
        #   # self.data_dir = os.path.join(opt.data_dir, '10499_test_img_new.txt')
        #   # self.data_dir = os.path.join(opt.data_dir, '10499_test_img_new.txt')
        #   # self.coord_dir = os.path.join(opt.data_dir, '10499_test_coord_newest_v2.txt')
        else:
            self.data_dir = os.path.join(opt.simsiam_dir, opt.test_img_txt)
            self.ind_dir = os.path.join(opt.simsiam_dir, opt.test_coord_txt)
            self.names_dir = names  
            self.name = name 
            # self.coord_dir = os.path.join(opt.data_dir, opt.test_coord_txt)
        #   # self.data_dir = os.path.join(opt.data_dir, '10499_test_img_new.txt')
        #   # self.coord_dir = os.path.join(opt.data_dir, '10499_test_coord_newest_v2.txt')
        #   # self.data_dir = os.path.join(opt.data_dir, '10304_new_test_img_all.txt')
        #   # self.coord_dir = os.path.join(opt.data_dir, '10304_val_coord_new.txt')
        # self.split = split 
        self.opt = opt 
        # self.size = size
        # self.coords = []
        # self.names_all = []
        # self.low = low 
        # self.up = up 
        # self.tomo_size = tomo_size
        self.split = split
        # self.hm_shape = None
        # self.sigma1 = sigma1 
        # self.sigma2 = sigma2
        # self.images, self.targets, self.names = self.load_data()
        # self.slice_range = slice_range
        # self.K = K
        self.indices = np.load(self.ind_dir)
        self.subvols = np.load(self.data_dir)
        self.names_all = np.load(self.names_dir)
        # self.subvols = self.subvols[np.where(self.names_all == self.name)]
        self.subvols_2d = self.subvols[:, 0:1,:,:]
        self.subvols_3d = self.subvols[:,1:2, :, :]
        self.mean_subvols_2d = np.mean(self.subvols_2d)
        self.mean_subvols_3d = np.mean(self.subvols_3d)
        self.subvols_2d = self.subvols_2d[np.where(self.names_all == self.name)]
        self.subvols_3d = self.subvols_3d[np.where(self.names_all == self.name)]
        # print('mean subvols', self.mean_subvols)
        self.indices = self.indices[np.where(self.names_all == self.name)]
        self.std_subvols_2d = np.std(self.subvols_2d)
        self.std_subvols_3d = np.std(self.subvols_3d)
        # print('std subvols', self.std_subvols)
        self.num_neighbors = num_neighbors
        if self.num_neighbors is not None:
            self.indices = self.indices[:,:self.num_neighbors+1]
        assert (self.indices.shape[0] == self.subvols_2d.shape[0])
        assert (self.indices.shape[0] == self.subvols_3d.shape[0])
        # print('subvols shape', self.subvols.shape)
        # self.tomos, self.names, self.subvols = self.load_data()
        # self.transforms = tio.Compose([
        #   tio.RandomBlur(std=(0,1),p=0.15),
        #   tio.RandomNoise(p=0.5),
        #   tio.RandomAffine(scales=(1,1,1,1,1,1), translation=(0,0,0,0,0,0,),degrees=(0,60,0,0,0,0),p=0.75),
        #   tio.transforms.Crop((self.size[0]//8, self.size[1]//8, self.size[2]//8)),
        #   tio.transforms.ZNormalization(),
        #   tio.transforms.RescaleIntensity(out_min_max=(-3,3)),
        #   tio.transforms.ZNormalization()
        #       ]
        #   )
        self.size = self.subvols.shape[1:]
        # self.transforms = T.Compose([
        #     T.ToPILImage(),
        #     T.RandomHorizontalFlip(0.5),
        #     T.RandomVerticalFlip(0.5),
        #     # T.RandomRotation(60),
        #     T.RandomAffine(degrees=30, translate=(0.1,0.2)),
        #     T.CenterCrop(self.size[1]-self.size[1]//4),
        #     T.ToTensor(),
        #     CornerErasing(p=0.5,scale = (0.01, 0.015), ratio = (0.5, 1.5)),]
        #     # CenterOut(crop_dim = 18),
        #     # T.Normalize((0.5),(0.5))]
        #     )
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            # T.GaussianBlur(3, sigma=(0.1,0.2)),
            T.RandomRotation(360),
            # T.RandomAffine(degrees=45, translate=(0.1,0.2)),
            # AdjustBrightness(),
            # InvertColor(),
            T.CenterCrop(opt.bbox),
            T.ToTensor(),
            CornerErasing(p=0.5,scale = (0.01, 0.02), ratio = (0.5, 1.5)),
            # CenterOut(crop_dim = 12),
            T.Normalize((self.mean_subvols_2d, self.mean_subvols_3d),(self.std_subvols_2d, self.std_subvols_3d))]
            )

        self.weak_transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(360),
            # T.RandomAffine(degrees=45, translate=(0.1,0.2)),
            T.CenterCrop(opt.bbox),
            T.ToTensor(),
            # CornerErasing(p=0.5,scale = (0.01, 0.02), ratio = (0.5, 1.5)),
            # CenterOut(crop_dim = 18),
            T.Normalize((self.mean_subvols_2d, self.mean_subvols_3d),(self.std_subvols_2d, self.std_subvols_3d))]
            )
        # self.weak_transforms = T.Compose([
        #     T.ToPILImage(),
        #     T.RandomHorizontalFlip(0.5),
        #     T.RandomVerticalFlip(0.5),
        #     # T.RandomRotation(60),
        #     # T.RandomAffine(degrees=30, translate=(0.1,0.2)),
        #     T.CenterCrop(self.size[1]-self.size[1]//4),
        #     T.ToTensor(),
        #     ]
        #     # CenterOut(crop_dim = 18),
        #     # T.Normalize((0.5),(0.5))]
        #     )

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

        tilt_images, recs, angles = load_tomo_all_and_angles_from_list(image_list.image_name, image_list.tilt_path, image_list.rec_path, image_list.angle_path, compress=False, denoise=self.opt.gauss)

      
        tomos = []
        names = []
        sub_vols = []
        vol_size = np.asanyarray(self.size)
        for k, v in tilt_images.items():
            names.append(k)
            angle = angles[k]
            rec = recs[k]
            topk_zs, topk_ys, topk_xs = get_potential_coords(rec, sigma1=self.sigma1, sigma2=self.sigma2,K = self.K)
            topk_zs, topk_ys, topk_xs = topk_zs.squeeze(), topk_ys.squeeze(), topk_xs.squeeze()
            positions = torch.stack((topk_xs, topk_ys, topk_zs), dim=1)
            num_of_coords = positions.shape[0]

            used_tilts = np.where((angle >= self.low) & (angle <= self.up))[0]
            used_angles = angle[np.where((angle >= self.low) & (angle <= self.up))]

            used_v = v[used_tilts]


            for p in range(num_of_coords):
                # patches = None 
                patches = []
                for ind,an in enumerate(used_angles):
                    tx, ty = self.convert_tomo_to_tilt(positions[p], an, self.tomo_size)
                    if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[2]//2 or ty >= self.tomo_size[1]-self.size[1]//2:
                        continue 

                    patch = used_v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2]

                    patches.append(patch)

                if len(patches) > 0:
                    shape = len(patches)
                    weight = self.gaussian_1d(shape)
                    patches = np.asarray(patches)
                    patches = np.average(patches, axis=0, weights = weight)
                    patches -= patches.min()

                    if patches.max() != 0:
                        patches /= patches.max()
                        patches = torch.tensor(patches)
                        patches = patches.unsqueeze(0).float()

                        sub_vols.append(patches)
                        self.coords.append(positions[p])
                        self.names_all.append(k)


            tomos.append(used_v)


        return tomos, names, sub_vols