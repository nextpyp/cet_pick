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
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr, CornerErasing, CenterOut,AdjustBrightness,InvertColor
import torchio as tio 
from multiprocessing import Pool


class TOMOPreProjAngleSelect3D(Dataset):
    num_classes = 1 
    default_resolution = [256, 256]

    def __init__(self, opt, split, size, low = -20, up = 20, tomo_size = (512, 512, 256), sigma1=2, sigma2=4, K=5000):
        super(TOMOPreProjAngleSelect3D, self).__init__()
        if split == 'train':
            self.data_dir = os.path.join(opt.data_dir, opt.train_img_txt)
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
            self.data_dir = os.path.join(opt.data_dir, opt.test_img_txt)
            # self.coord_dir = os.path.join(opt.data_dir, opt.test_coord_txt)
        #   # self.data_dir = os.path.join(opt.data_dir, '10499_test_img_new.txt')
        #   # self.coord_dir = os.path.join(opt.data_dir, '10499_test_coord_newest_v2.txt')
        #   # self.data_dir = os.path.join(opt.data_dir, '10304_new_test_img_all.txt')
        #   # self.coord_dir = os.path.join(opt.data_dir, '10304_val_coord_new.txt')
        # self.split = split 
        self.opt = opt 
        self.size = size
        self.crop_size_x, self.crop_size_y = int(np.ceil(self.size[1]*np.sqrt(2))), int(np.ceil(self.size[2]*np.sqrt(2)))
        self.coords = []
        self.names_all = []
        self.low = low 
        self.up = up 
        self.tomo_size = tomo_size
        self.split = split
        self.hm_shape = None
        self.sigma1 = sigma1 
        self.sigma2 = sigma2
        # self.images, self.targets, self.names = self.load_data()
        # self.slice_range = slice_range
        self.K = K
        self.tomos, self.names, self.subvols, self.subvols_sets, self.sub_vols_3d, self.sub_vols_sets_3d = self.load_data()
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

        self.transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            # T.GaussianBlur(3, sigma=(0.1,0.2)),
            # T.RandomRotation(90),
            T.RandomAffine(degrees=45, translate=(0.1,0.2)),
            # AdjustBrightness(),
            # InvertColor(),
            T.CenterCrop(self.size[1]),
            T.ToTensor(),
            # CornerErasing(p=0.3,scale = (0.01, 0.02), ratio = (0.5, 1.5)),
            # CenterOut(crop_dim = 12),
            T.Normalize((self.mean_subvols, self.mean_subvols3d),(self.std_subvols, self.std_subvols3d))]
            )
        # print('mean_subvol3d', self.mean_subvols3d)
        # print('std_subvols3d', self.std_subvols3d)
        # print('mean subvol', self.mean_subvols)
        # print('std subvol', self.std_subvols)

        self.weak_transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(360),
            # T.RandomAffine(degrees=45, translate=(0.1,0.2)),
            T.CenterCrop(self.size[1]),
            T.ToTensor(),
            CornerErasing(p=0.5,scale = (0.01, 0.02), ratio = (0.5, 1.5)),
            # CenterOut(crop_dim = 24),
            T.Normalize((self.mean_subvols3d),(self.std_subvols3d))]
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

    def extract_3d_tomo(self, rec, tomo_coord):
        x, y, z = tomo_coord[0], tomo_coord[1], tomo_coord[2]

        # patch_3d = rec[z, y - self.size[1]//2:y + self.size[1]//2, x - self.size[1]//2:x + self.size[1]//2].copy()
        patch_3d = rec[z, y - self.crop_size_y//2:y + self.crop_size_y//2, x - self.crop_size_x//2:x + self.crop_size_x//2].copy()
        patch_3d = (patch_3d - np.min(patch_3d))/(np.max(patch_3d) - np.min(patch_3d))
        # patch_3d -= patch_3d.min()
        # patch_3d /= patch_3d.max()
        patch_3d = torch.tensor(patch_3d)
        patch_3d = patch_3d.unsqueeze(0).float()
        return patch_3d

    def extract_patches(self, v, tomo_coord, angles, tomo_size=[512, 512, 256]):
        patches = None 
        for ind, an in enumerate(angles):
            tx, ty = self.convert_tomo_to_tilt(tomo_coord, an, tomo_size)
            # if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[1]//2 or ty >= self.tomo_size[1]-self.size[1]//2:
            #             continue 
            # patch = v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2].copy()
            if tx <= self.crop_size_x//2+10 or tx >= self.tomo_size[1]-self.crop_size_x//2-10 or ty <= self.crop_size_y//2+10 or ty >= self.tomo_size[1]-self.crop_size_y//2-10:
                        continue 
            patch = v[ind, ty - self.crop_size_y//2: ty + self.crop_size_y//2, tx - self.crop_size_x//2: tx + self.crop_size_x//2].copy()
            if patches is None:
                patches = patch 
            else:
                patches += patch 
        if patches is not None:
            if np.min(patches) != np.max(patches):
                patches = (patches - np.min(patches))/(np.max(patches) - np.min(patches))

            # patches -= patches.min()
            # if patches.max() != 0:
                # patches /= patches.max()
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
        # coords = pd.read_csv(self.coord_dir, sep='\t')
        # images = load_tomos_from_list(image_list.image_name, image_list.path, order='zxy', compress=False, denoise=False, tilt=True)
        # images, angles = load_tomos_and_angles_from_list(image_list.image_name, image_list.path, image_list.angle_path, order='zxy', compress=False, denoise=self.opt.gauss, tilt=True)
        # print('self.opt.gauss', self.opt.gauss)
        tilt_images, recs, angles = load_tomo_all_and_angles_from_list(image_list.image_name, image_list.tilt_path, image_list.rec_path, image_list.angle_path, compress=False, denoise=self.opt.gauss)

        # if self.split == 'train':
        #   x_s, y_s, z_s = np.arange(self.size[1]//2, self.tomo_size[0]-self.size[1]//2, 2), np.arange(self.size[1]//2, self.tomo_size[0]-self.size[1]//2, 2), np.arange(50, 180, 2)
        #   xv, yv, zv = np.meshgrid(x_s, y_s, z_s, indexing = 'xy')
        #   positions = np.vstack([xv.ravel(), yv.ravel(),zv.ravel()])
        #   # num_particles = len(coords)
        #   num_of_coords = positions.shape[-1]
        # elif self.split == 'test':

        #   # x_s, y_s, z_s = np.arange(self.size[1]//2, self.tomo_size[0]-self.size[1]//2, 1), np.arange(self.size[1]//2, self.tomo_size[0]-self.size[1]//2, 1), np.arange(self.slice_range[0], self.slice_range[1], 2)
        #   # xv, yv, zv = np.meshgrid(x_s, y_s, z_s, indexing = 'xy')
        #   # positions = np.vstack([xv.ravel(), yv.ravel(),zv.ravel()])
        #   # num_particles = len(coords)
        #   topk_zs, topk_ys, topk_xs = get_potential_coords()
        #   print('all_pos', positions)
        #   num_of_coords = positions.shape[-1]
        # if 
        # else:
        #   ims, targets, names = self.match_images_targets(images, coords)
        # num_of_tomos = len(ims)
        # print('num_of_tomos', num_of_tomos)
        # print('coords', coords)
        tomos = {}
        names = []
        sub_vols = []
        sub_vols_3d = []
        sub_vols_sets = []
        sub_vols_sets_3d = []
        # coords = []
        self.angles = {}
        # from_tomo = []
        vol_size = np.asanyarray(self.size)
        # rec_info = {}
        for k, v in tilt_images.items():
            names.append(k)
            angle = angles[k]
            rec = recs[k]
            np.save('/nfs/bartesaghilab/qh36/3D_picking/cet_pick_github/cet_pick/cet_pick/used_rec.npy', rec)
            # tomos[k] = rec
            print('rec', rec.shape)
            print('name', k)
            # topk_zs, topk_ys, topk_xs = get_potential_coords(rec, sigma1=self.sigma1, sigma2=self.sigma2,K = self.K)
            topk_zs, topk_ys, topk_xs = get_potential_coords_pyramid(rec, sigma_init = self.sigma1, K = self.K)
            topk_zs, topk_ys, topk_xs = topk_zs.squeeze(), topk_ys.squeeze(), topk_xs.squeeze()
            # print('coords', topk_zs, topk_ys, topk_xs)
            positions = torch.stack((topk_xs, topk_ys, topk_zs), dim=1)
            num_of_coords = positions.shape[0]
            print('num_of_coords', num_of_coords)
            # print('angle', angle)
            used_tilts = np.where((angle >= self.low) & (angle <= self.up))[0]
            used_angles = angle[np.where((angle >= self.low) & (angle <= self.up))]
            used_v = v[used_tilts]
            self.angles[k] = used_angles
            for p in range(num_of_coords):
                patches = None 
                curr_coord = positions[p]
                if p == 100:
                    print('curr_coord', curr_coord)
                # offs_x_y = [-2, -1, 0, 1, 2]
                # offs_z = [-1, 0, 1]
                # off_x = random.sample(offs_x_y, 1)[0]
                # off_y = random.sample(offs_x_y, 1)[0]
                # off_z = random.sample(offs_z, 1)[0]
                tomo_x, tomo_y, tomo_z = positions[p][0], positions[p][1], positions[p][2]
                # print('curr coord', positions[p])

                x_c_1, y_c_1, z_c_1 = tomo_x, tomo_y, tomo_z + 1
                x_c_2, y_c_2, z_c_2 = tomo_x, tomo_y, tomo_z - 1
                x_c_3, y_c_3, z_c_3 = tomo_x-1, tomo_y, tomo_z - 1
                x_c_4, y_c_4, z_c_4 = tomo_x, tomo_y+1, tomo_z - 1
                aug_all = [[x_c_1, y_c_1, z_c_1],[x_c_2, y_c_2, z_c_2],[x_c_3, y_c_3, z_c_3],[x_c_4, y_c_4, z_c_4]]
                # print('off coord', [x_c_r, y_c_r, z_c_r])
                patch_orig = self.extract_patches(used_v, curr_coord, used_angles)
                if not patch_orig is None:
                    patch_orig_3d = self.extract_3d_tomo(rec, curr_coord)
                    if p == 100:
                        np.save('/nfs/bartesaghilab/qh36/3D_picking/cet_pick_github/cet_pick/cet_pick/patch_100.npy', patch_orig_3d.numpy())
                        np.save('/nfs/bartesaghilab/qh36/3D_picking/cet_pick_github/cet_pick/cet_pick/patch_100_p.npy', patch_orig.numpy())
                    if self.split == 'train':
                    # print('orig is not none')
                        patch_sets = [patch_orig]
                        patch3d_sets = [patch_orig_3d]
                        for aug_ind in aug_all:
                            patch_aug = self.extract_patches(used_v, aug_ind, used_angles)
                            if patch_aug is None:
                                continue 
                            patch_aug_3d = self.extract_3d_tomo(rec, aug_ind)
                            patch_sets.append(patch_aug)
                            patch3d_sets.append(patch_aug_3d)

                        # patch_aug = self.extract_patches(used_v, [x_c_r, y_c_r, z_c_r], used_angles)
                        if len(patch_sets) > 1:
                            # print('aug is not none')
                            sub_vols_3d.append(patch_orig_3d)
                            sub_vols_sets_3d.append(patch3d_sets)
                            sub_vols_sets.append(patch_sets)
                            sub_vols.append(patch_orig)
                            self.coords.append(positions[p])
                            self.names_all.append(k)
                    else:
                        # subvol_3d = self.extract_3d_tomo(rec, positions[p])
                        sub_vols.append(patch_orig)
                        sub_vols_3d.append(patch_orig_3d)
                        self.coords.append(positions[p])
                        self.names_all.append(k)


                # for ind,an in enumerate(used_angles):
                #     tx, ty = self.convert_tomo_to_tilt(positions[p], an, self.tomo_size)
                #     tx1, ty1 = self.convert_tomo_to_tilt([x_c_r, y_c_r, z_c_r], an, self.tomo_size)
                #     if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[2]//2 or ty >= self.tomo_size[1]-self.size[1]//2:
                #         continue 

                #     patch = used_v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2]
                #     if patches is None:
                #         patches = patch 
                #     else:
                #         patches += patch 

                    

                #     if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[2]//2 or ty >= self.tomo_size[1]-self.size[1]//2:
                #         continue 

                #     patch = used_v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2]
                #     if patches is None:
                #       patches = patch 
                #     else:
                #       patches += patch 

                # if not patches is None:
                #     patches -= patches.min()
                #     if patches.max() != 0:
                #         patches /= patches.max()
                #         patches = torch.tensor(patches)
                #         patches = patches.unsqueeze(0).float()
                        

            tomos[k] = used_v
        self.mean_subvols3d = torch.mean(torch.stack(sub_vols_3d))
        self.std_subvols3d = torch.std(torch.stack(sub_vols_3d))
        self.mean_subvols = torch.mean(torch.stack(sub_vols))
        self.std_subvols = torch.std(torch.stack(sub_vols))
            

        return tomos, names, sub_vols, sub_vols_sets, sub_vols_3d, sub_vols_sets_3d