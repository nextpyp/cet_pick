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


class TOMOSCANProjAngleSelect(Dataset):
    num_classes = 1 
    default_resolution = [256, 256]

    def __init__(self, opt, split, num_neighbors=None):
        super(TOMOSCANProjAngleSelect, self).__init__()
        if split == 'train':
            self.data_dir = os.path.join(opt.simsiam_dir, opt.train_img_txt)
            self.ind_dir = os.path.join(opt.simsiam_dir, opt.train_coord_txt)
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
        self.mean_subvols = np.mean(self.subvols)
        print('mean subvols', self.mean_subvols)

        self.std_subvols = np.std(self.subvols)
        print('std subvols', self.std_subvols)
        self.num_neighbors = num_neighbors
        if self.num_neighbors is not None:
            self.indices = self.indices[:,:self.num_neighbors+1]
        assert (self.indices.shape[0] == self.subvols.shape[0])
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
            # T.RandomRotation(30),
            T.RandomAffine(degrees=45, translate=(0.1,0.2)),
            # AdjustBrightness(),
            # InvertColor(),
            T.CenterCrop(self.size[1]-self.size[1]//4),
            T.ToTensor(),
            CornerErasing(p=0.5,scale = (0.01, 0.02), ratio = (0.5, 1.5)),
            # CenterOut(crop_dim = 12),
            T.Normalize((self.mean_subvols),(self.std_subvols))]
            )

        self.weak_transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            # T.RandomRotation(30),
            # T.RandomAffine(degrees=45, translate=(0.1,0.2)),
            T.CenterCrop(self.size[1]-self.size[1]//4),
            T.ToTensor(),
            # CornerErasing(p=0.5,scale = (0.01, 0.02), ratio = (0.5, 1.5)),
            # CenterOut(crop_dim = 18),
            T.Normalize((self.mean_subvols),(self.std_subvols))]
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
        tomos = []
        names = []
        sub_vols = []
        vol_size = np.asanyarray(self.size)
        for k, v in tilt_images.items():
            names.append(k)
            angle = angles[k]
            rec = recs[k]
            print('rec', rec.shape)
            topk_zs, topk_ys, topk_xs = get_potential_coords(rec, sigma1=self.sigma1, sigma2=self.sigma2,K = self.K)
            topk_zs, topk_ys, topk_xs = topk_zs.squeeze(), topk_ys.squeeze(), topk_xs.squeeze()
            positions = torch.stack((topk_xs, topk_ys, topk_zs), dim=1)
            num_of_coords = positions.shape[0]
            print('num_of_coords', num_of_coords)
            # print('angle', angle)
            used_tilts = np.where((angle >= self.low) & (angle <= self.up))[0]
            used_angles = angle[np.where((angle >= self.low) & (angle <= self.up))]
            # print(used_angles)
            # print(used_tilts)
            used_v = v[used_tilts]

            
            # with Pool() as pool:
            #   args = [(used_v, positions[:,p], used_angles) for p in range(num_of_coords)]
            #   results = pool.starmap(self.extract_patches, args)
            #   print(results)
            print('tomo size', self.tomo_size)
            print('size', self.size)
            for p in range(num_of_coords):
                # patches = None 
                patches = []
                # if positions[p][-1] == 128:
                #   print(positions[p])
                #   print('positions hey', positions[:,p])
                for ind,an in enumerate(used_angles):
                    tx, ty = self.convert_tomo_to_tilt(positions[p], an, self.tomo_size)
                    # if positions[:,p][1] > 240:
                    #   print(tx, ty)

                        # print('condition1', tx <= self.size[1]//2)
                        # print('condition2,', tx >= self.tomo_size[1]-self.size[1]//2)
                        # print('conditino3', ty <= self.size[2]//2)
                        # print('condition4',  ty >= self.tomo_size[2]-self.size[1]//2)
                    if tx <= self.size[1]//2 or tx >= self.tomo_size[1]-self.size[1]//2 or ty <= self.size[2]//2 or ty >= self.tomo_size[1]-self.size[1]//2:
                        continue 

                    patch = used_v[ind, ty - self.size[2]//2: ty + self.size[2]//2, tx - self.size[1]//2: tx + self.size[1]//2]
                    # if positions[:,p][1] > 240:
                    #   print('patch', patch.shape)
                    patches.append(patch)
                    # if patches is None:
                    #   patches = patch 
                    # else:
                    #   patches += patch 
                # print('patch', patch.shape)
                # if not patches is None:
                if len(patches) > 0:
                    shape = len(patches)
                    weight = self.gaussian_1d(shape)
                    patches = np.asarray(patches)
                    patches = np.average(patches, axis=0, weights = weight)
                    patches -= patches.min()
                    # if positions[:,p][1] > 240:
                    #   # print('positions hey', positions[:,p])
                    #   print('patches max', patches.max())
                    if patches.max() != 0:
                        patches /= patches.max()
                        patches = torch.tensor(patches)
                        patches = patches.unsqueeze(0).float()
                        # print('patches', patches.shape)
                        sub_vols.append(patches)
                        # if positions[:,p][1] > 240:
                        #   print('positions', positions[:,p])
                        # print('positions,', positions[:,p])
                        self.coords.append(positions[p])
                        self.names_all.append(k)

            # print('v', v.shape)
            # v = v[17:32]
            # v = v[19:22]
            tomos.append(used_v)
            # blks = cutup(v, self.size ,(1,1,1))
            # self.hm_shape = blks.shape

            # for i in range(blks.shape[0]):
            #   for j in range(blks.shape[1]):
            #       for k in range(blks.shape[2]):
            #           mean_blk = np.mean(blks[i,j,k], axis=0)
            #           mean_blk -= mean_blk.min()
            #           # print('max',mean_blk.max())
            #           # print('min',mean_blk.min())
            #           self.coords.append(torch.tensor([i,j,k]))
                        
            #           if mean_blk.max() != 0:
            #               mean_blk /= mean_blk.max()
            #               # print(mean_blk.shape)
            #               mean_blk = torch.tensor(mean_blk)
            #               mean_blk = mean_blk.unsqueeze(0).float()
            #               # print(mean_blk.shape)
            #               # mean_blk = torch.tensor()
            #               sub_vols.append(mean_blk)
            # del blks
            # blks = blks.reshape(-1, *vol_size)
            # for p in blks:
            #   sub_vols.append(p)


        # hms = []
        # inds = []
        # gt_dets = []
        # for i in range(num_of_tomos):
        #   tomo = ims[i]
        #   coords = targets[i]
        #   depth, height, width = tomo.shape[0], tomo.shape[1], tomo.shape[2]
        #   num_objs = len(coords)
        #   # print('num_objs', num_objs)
        #   output_h, output_w = height // self.opt.down_ratio, width // self.opt.down_ratio
        #   num_class = self.num_classes
        #   # if self.opt.pn:
        #   #   hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
        #   # else:
        #   hm = np.zeros((depth, output_h, output_w), dtype=np.float32)
        #   ind = np.zeros((num_objs), dtype=np.int64)
        #   draw_gaussian = draw_umich_gaussian_3d
        #   gt_det = []
        #   # all_coords = [np.arange(output_h), np.arange(output_w), np.arange(width)]
        #   h = self.opt.bbox // self.opt.down_ratio
        #   for k in range(num_objs):
        #       ann = coords[k]
        #       radius = gaussian_radius((math.ceil(h), math.ceil(h)))
        #       radius = max(0, int(radius))
        #       ann = self._downscale_coord(ann)
        #       ann = np.array(ann)
        #       ct_int = ann.astype(np.int32)
        #       z_coord = ct_int[-1]
        #       draw_gaussian(hm, ct_int, radius,  0, 0, 0, discrete=False)
        #       ind[k] = ct_int[2] * (output_w * output_h) + ct_int[1] * output_w + ct_int[0]
        #       # print('ann', ann)
        #       gt_det.append(ann)
        #   gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
        # # all_coords = np.array(all_coords)
        #   tomos.append(tomo)
        #   hms.append(hm)
        #   inds.append(ind)
        #   gt_dets.append(gt_det)
        # gt_dets = [gt_det]

        return tomos, names, sub_vols