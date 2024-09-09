from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from cet_pick.models.model import create_model, load_model 
from cet_pick.utils.debugger import Debugger 
from cet_pick.utils.loader import load_rec

class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv,last_k = opt.last_k)
        self.model = load_model(self.model, opt.load_model)
        if opt.task == 'semiclass':
            self.model.fill()
        self.model = self.model.to(opt.device)
        self.model.eval()



        self.max_per_image = 900
        self.opt = opt 
        self.pause = True 

    def pre_process(self, tomo, meta=None):
        height, width, depth = tomo.shape[0:3]
        c = np.array([width // 2, height // 2], dtype=np.float32)
        s = np.array([width, height], dtype = np.float32)
        tomo = torch.from_numpy(images)
        meta = {'c': c, 's': s, 'out_height': height // self.opt.down_ratio, 'out_width': width // self.opt.down_ratio, 'zdim': depth}
        return images

    def process(self, images, return_time=False):
        raise NotImplementedError 

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def save_detection(self, dets, path, meta, prefix='', name=''):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0 
        debugger = Debugger(dataset = self.opt.dataset)
        start_time = time.time()
        pre_processed = False 

        loaded_time = time.time()
        load_time += (loaded_time - start_time)
        detections = []

        images = image_or_path_or_tensor
        if self.opt.task != 'semiclass':
            images = images.to(self.opt.device, non_blocking=True)

        # torch.cuda().synchronize()
        pre_process_time = time.time()
        pre_time += 0
        # out_hm = self.process(images, return_time=True)
        # return out_hm
        output, dets, hm, forward_time = self.process(images, return_time=True)
        # return dets, hm
        batch, cat, depth, height, width = hm.size()
        # torch.cuda.synchronize()
        net_time += forward_time - pre_process_time
        decode_time = time.time()
        dec_time += decode_time - forward_time

        if self.opt.debug >= 2:
            self.debug(debugger, images, dets, output)

        dets, name = self.post_process(dets, meta, z_dim_tot=depth)
        # print('dets', dets)
        # print('name', name)
        if self.opt.gpus[0] >= 0: 
            torch.cuda.synchronize()
        post_process_time = time.time()
        post_time += post_process_time - decode_time 
        self.save_detection(hm, dets, self.opt.out_path,meta, name = name)

        # torch.cuda.synchronize()
        end_time = time.time()
        tot_time += end_time - start_time
        #   self.show_results(debugger, image, results)
        return {'tot_time': tot_time, 'load': load_time, 'pre': pre_time, 'net': net_time, 'dec': dec_time}




