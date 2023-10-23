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
from cet_pick.detectors.base_detector import BaseDetector
from cet_pick.utils.post_process import tomo_post_process

class TomodetDetector(BaseDetector):
    def __init__(self, opt):
        super(TomodetDetector, self).__init__(opt)
        # self.opt = opt 

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            # wh = output['wh']
            # reg = output['reg'] if self.opt.reg_offset else None 
            if self.opt.gpus[0] >= 0:
                torch.cuda.synchronize()
            forward_time = time.time()
            dets = tomo_decode(hm, reg=None, K = self.opt.K)

        if return_time:
            return output, dets, hm, forward_time
        else:
            return output, dets, hm

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        # print('dets,', dets.shape)
        # dets[:,:,:2] *= self.opt.down_ratio
        dets[:,:,:2] *= self.opt.down_ratio
        # dets = dets.reshape(1, -1, dets.shape[2])
        post_dets = tomo_post_process(dets, z_dim_tot=128)
        preds = post_dets[0]
        name = meta['name'][0]

        return preds, name

    def save_detection(self, hm, dets, path, prefix='', name=''):
        if not os.path.exists(path):
            os.mkdir(path)

        out_detect = open(path+'/{}.txt'.format(name), 'w+')
        hm = hm.detach().cpu().numpy()[0][0]
        # print('hm', hm.shape)
        hm = np.swapaxes(hm, 1, 0)
        out_hm = path+'/{}_hm.mrc'.format(name)
        if(np.isnan(hm).any()):
        # print("The Array contain NaN values")
            raise ValueError("Output contains NaN values")
        with mrcfile.new(out_hm, overwrite=True) as mrc:
            mrc.set_data(np.float32(hm))
        # print('x_coord' + '\t' + 'z_coord' + '\t' + 'y_coord' + '\t' + 'score', file=out_detect)
        for k, v in dets.items():
            for c in v:
                x, y, z, score = int(np.floor(c[0])), int(np.floor(c[1])), int(np.floor(c[2])), float(c[3])
                # x, y, z, score = int(np.floor(c[0])), int(np.floor(c[1])), int(np.floor(c[2])), float(c[3])
                conf = c[4]

                if score > self.opt.thresh and z > 35 and z < 95 and x > 20 and x < 500 and y > 20 and y < 500:
                    # print(str(x) + '\t' + str(z) + '\t' + str(y) + '\t' + str(score), file = out_detect)
                    z = int(z*2)
                    if not self.opt.with_score:
                        print(str(x) + '\t' + str(z) + '\t' + str(y),file = out_detect)
                    else:
                        print(str(x) + '\t' + str(z) + '\t' + str(y) + '\t' + str(score), file = out_detect)

    def merge_outputs(self, detections):
        scores = detectors[:, -1]
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image 
            thresh = np.partition(scores, kth)[kth]
            keep_inds = (scores >= thresh)
            detections = detections[keep_inds]

        return detections

    def debug(self, debugger, images, dets, output, scale = 1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:,:,:2] *= self.opt.down_ratio 
        for i in range(1):
            img = images[i].detach().cpu().numpy()
            for k in range(len(dets[i])):
                z_slice = dets[i,k,2]
                pred = debugger.gen_colormap(output['hm'][i][k].detach().cpu().numpy())
                im = img[z_slice,:,:]
                debugger.add_blend_img(im, pred, 'pred_hm_{:.1f}'.format(scale))
                debugger.add_slice(img, slice_num=z_slice)
                if detection[i,k,4] > self.opt.center_thresh:
                    debugger.add_circle(detection[i,k,:2], detection[i,k,3], detection[i,k,2])

    def show_results(self, debugger, image, results):
        for j in range(image.shape[0]):
            debugger.add_slice(image, j)
            for item in detections:
                pass
                # if item[]

