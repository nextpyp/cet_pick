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
from cet_pick.models.decode import tomo_decode 
from cet_pick.utils.debugger import Debugger 
from cet_pick.models.utils import _sigmoid 
from cet_pick.detectors.base_detector import BaseDetector
from cet_pick.utils.post_process import tomo_post_process, tomo_fiber_postprocess

class TomodetDetector(BaseDetector):
    def __init__(self, opt):
        super(TomodetDetector, self).__init__(opt)
        # self.opt = opt 

    def process(self, images, return_time=False):
        with torch.no_grad():

            output = self.model(images)[-1]
            hm = output['hm']

            if self.opt.gpus[0] >= 0:
                torch.cuda.synchronize()
            forward_time = time.time()

            hm = _sigmoid(hm)
           

            dets = tomo_decode(hm, kernel = self.opt.nms, reg=None, K = self.opt.K, if_fiber = self.opt.fiber)
        if return_time:
            return output, dets, hm, forward_time
        else:
            return output, dets, hm

    def post_process(self, dets, meta, scale=1, z_dim_tot=128):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:,:,:2] *= self.opt.down_ratio

        post_dets = tomo_post_process(dets, z_dim_tot=z_dim_tot)

        preds = post_dets[0]
        name = meta['name'][0]

        return preds, name

    def save_detection(self, hm, dets, path, meta, prefix='', name=''):
        if not os.path.exists(path):
            os.mkdir(path)

        out_detect = open(path+'/{}.txt'.format(name), 'w+')
        hm = hm.detach().cpu().numpy()[0][0]
        max_z, max_y, max_x = hm.shape
        max_x, max_y = max_x*2, max_y*2
        hm = np.swapaxes(hm, 1, 0)
        out_hm = path+'/{}_hm.mrc'.format(name)
        pre_coords = []
        if(np.isnan(hm).any()):
            raise ValueError("Output contains NaN values")
        with mrcfile.new(out_hm, overwrite=True) as mrc:
            mrc.set_data(np.float32(hm))
        for k, v in dets.items():
            for c in v:
                x, y, z, score = int(np.floor(c[0])), int(np.floor(c[1])), int(np.floor(c[2])), float(c[3])
                conf = c[4]

                if (score > self.opt.out_thresh and z >= self.opt.cutoff_z and z <= max_z-self.opt.cutoff_z and x > 20 and x < max_x - 20 and y > 20 and y < max_y - 20):
                    if self.opt.compress:
                        z = int(z)*2
                    if self.opt.fiber:
                        pre_coords.append([x,y,z])

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
               

