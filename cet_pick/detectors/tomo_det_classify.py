from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os
import mrcfile
from cet_pick.models.decode import tomo_decode, tomo_decode_classify 
from cet_pick.utils.debugger import Debugger 
from cet_pick.models.utils import _sigmoid 
from cet_pick.detectors.base_detector import BaseDetector
from cet_pick.utils.post_process import tomo_post_process, tomo_fiber_postprocess, tomo_group_postprocess

class PatchDataset:
    def __init__(self,tomo, patch_size_z=48, patch_size_xy = 96, padding_z=12, padding_xy = 24):
        self.tomo = tomo
        self.patch_size_xy = patch_size_xy
        self.patch_size_z = patch_size_z
        self.padding_xy = padding_xy
        self.padding_z = padding_z

        nz,ny,nx = tomo.shape

        pz = int(np.ceil(nz/patch_size_z))
        py = int(np.ceil(ny/patch_size_xy))
        px = int(np.ceil(nx/patch_size_xy))
        self.shape = (pz,py,px)
        self.num_patches = pz*py*px


    def __len__(self):
        return self.num_patches

    def __getitem__(self, patch):
        # patch index
        i,j,k = np.unravel_index(patch, self.shape)

        patch_size_xy = self.patch_size_xy
        patch_size_z = self.patch_size_z
        padding_xy = self.padding_xy
        padding_z = self.padding_z
        tomo = self.tomo

        # pixel index
        i = patch_size_z*i
        j = patch_size_xy*j
        k = patch_size_xy*k

        # make padded patch
        d_xy = patch_size_xy + 2*padding_xy
        d_z = patch_size_z + 2*padding_z
        x = np.zeros((d_z, d_xy, d_xy), dtype=np.float32)

        # index in tomogram
        si = max(0, i-padding_z)
        ei = min(tomo.shape[0], i+patch_size_z+padding_z)
        sj = max(0, j-padding_xy)
        ej = min(tomo.shape[1], j+patch_size_xy+padding_xy)
        sk = max(0, k-padding_xy)
        ek = min(tomo.shape[2], k+patch_size_xy+padding_xy)

        # index in crop
        sic = padding_z - i + si
        eic = sic + (ei - si)
        sjc = padding_xy - j + sj
        ejc = sjc + (ej - sj)
        skc = padding_xy - k + sk
        ekc = skc + (ek - sk)

        x[sic:eic,sjc:ejc,skc:ekc] = tomo[si:ei,sj:ej,sk:ek]
        return np.array((i,j,k), dtype=int),x

class TomoClassdetDetector(BaseDetector):
    def __init__(self, opt):
        super(TomoClassdetDetector, self).__init__(opt)
        self.opt = opt 

    def process(self, images, return_time=False):
        out_hm = torch.zeros_like(images, device = self.opt.device)
        # print('images', images.shape)
        if images.shape[0] <= 85 and images.shape[1] <= 128 and images.shape[2] <= 128:
            patch_size_z = 0
            patch_size_xy = 0
        else:
            patch_size_z = 32
            patch_size_xy = 96
            padding_z = 16
            padding_xy = 24
        if patch_size_z == 0:

            with torch.no_grad():

                output = self.model(images)[-1]
                hm = output['hm']

                if self.opt.gpus[0] >= 0:
                    torch.cuda.synchronize()
                forward_time = time.time()

                hm = _sigmoid(hm)
            out_hm = hm[0]
            out_hm[:,:,:30,:] = 0
            out_hm[:,:,-30:,:] = 0
            out_hm[:,:,:,:30] = 0
            out_hm[:,:,:,-30:] = 0
            detections = tomo_decode_classify(out_hm, self.opt.nms, self.opt.out_thresh)

            out_hm = out_hm.unsqueeze(0)
            output = None   

                
        else:
            patch_data = PatchDataset(images[0], patch_size_z, patch_size_xy, padding_z, padding_xy)
            batch_iterator = torch.utils.data.DataLoader(patch_data, batch_size=1)
            total = len(patch_data)
            count = 0
            forward_time = time.time()
            with torch.no_grad():
                for index, x in batch_iterator:
                    x = x.to(self.opt.device, non_blocking=True)

                    x = self.model(x)[-1]

                    x = x['hm']

                    if self.opt.gpus[0] >= 0:
                        torch.cuda.synchronize()
                    x = _sigmoid(x)

                    for b in range(x.shape[0]):
                        i,j,k = index[b]
                        xb = x[b][0]
                        patch = out_hm[0,i:i+patch_size_z, j:j+patch_size_xy, k:k+patch_size_xy]
                        pz,py,px = patch.shape
                        xb = xb[padding_z:padding_z+pz, padding_xy:padding_xy+py, padding_xy:padding_xy+px]
                        out_hm[0, i:i+patch_size_z, j:j+patch_size_xy, k:k+patch_size_xy] = xb  
                        count += 1
            out_hm[:,:,:30,:] = 0
            out_hm[:,:,-30:,:] = 0
            out_hm[:,:,:,:30] = 0
            out_hm[:,:,:,-30:] = 0
            detections = tomo_decode_classify(out_hm, self.opt.nms, self.opt.out_thresh)

            out_hm = out_hm.unsqueeze(0)


            output = None

        if return_time:
            return output, detections, out_hm, forward_time
        else:
            return output, detections, out_hm

    def post_process(self, dets, meta, scale=1, z_dim_tot=128):
        # dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        # dets[:,:,:2] *= self.opt.down_ratio
        # print(dets)
        dets[:,:2] *= self.opt.down_ratio
        # print('new_dets', dets)

        # post_dets = tomo_post_process(dets, z_dim_tot=z_dim_tot)

        # preds = post_dets[0]
        name = meta['name'][0]

        return dets, name

    def save_detection(self, hm, dets, path, meta, prefix='', name=''):
        if not os.path.exists(path):
            os.mkdir(path)

        out_detect = open(path+'/{}.txt'.format(name), 'w+')
        hm = hm.detach().cpu().numpy()[0][0]
        # print('hm', hm.shape)
        max_z, max_y, max_x = hm.shape
        # max_x, max_y = max_x*2, max_y*2
        hm = np.swapaxes(hm, 1, 0)
        out_hm = path+'/{}_hm.mrc'.format(name)
        pre_coords = []
        if(np.isnan(hm).any()):
            raise ValueError("Output contains NaN values")
        with mrcfile.new(out_hm, overwrite=True) as mrc:
            mrc.set_data(np.float32(hm))
        # for k, v in dets.items():
        for c in dets:
            x, y, z, score = int(np.floor(c[0])), int(np.floor(c[1])), int(np.floor(c[2])), np.float(c[3])
            conf = c[3]

            if (score > self.opt.out_thresh and z >= self.opt.cutoff_z and z <= max_z-self.opt.cutoff_z and x > 20 and x < max_x - 20 and y > 20 and y < max_y - 20):
                if self.opt.compress:
                    z = int(z)*2
                if self.opt.fiber:
                    pre_coords.append([x,y,z])
                elif self.opt.spike:
                    pre_coords.append([x,y,z,score])

                else:
                    if not self.opt.with_score:
                        print(str(x) + '\t' + str(z) + '\t' + str(y),file = out_detect)
                    else:
                        print(str(x) + '\t' + str(z) + '\t' + str(y) + '\t' + str(score), file = out_detect)
        if self.opt.fiber:
            post_coords = tomo_fiber_postprocess(pre_coords, distance_cutoff=self.opt.distance_cutoff, res_cutoff = self.opt.r2_cutoff, curvature_cutoff=self.opt.curvature_cutoff)
            for c in post_coords:
                print(str(c[0]) + '\t' + str(c[1]) + '\t' + str(c[2]),file = out_detect)
        if self.opt.spike:
            post_coords = tomo_group_postprocess(pre_coords, distance_cutoff=self.opt.distance_cutoff, min_per_group=5)
            for c in post_coords:
                if not self.opt.with_score:
                    print(str(c[0]) + '\t' + str(c[2]) + '\t' + str(c[1]),file = out_detect)
                else:
                    print(str(c[0]) + '\t' + str(c[2]) + '\t' + str(c[1]) + '\t' + str(c[3]),file = out_detect)


    def merge_outputs(self, detections):
        scores = detectors[:, -1]
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image 
            thresh = np.partition(scores, kth)[kth]
            keep_inds = (scores >= thresh)
            detections = detections[keep_inds]

        return detections

    def debug(self, debugger, images, dets, output, scale = 1):
        pass


    def show_results(self, debugger, image, results): #TODO
        for j in range(image.shape[0]):
            debugger.add_slice(image, j)
            for item in detections:
                pass
               

