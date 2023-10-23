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
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
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
        meta = {'c': c, 's': s, 'out_height': height // self.opt.down_ratio, 'out_width': width // self.opt.down_ratio}
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

    def save_detection(self, dets, path, prefix='', name=''):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0 
        debugger = Debugger(dataset = self.opt.dataset)
        start_time = time.time()
        pre_processed = False 
        # if isinstance(image_or_path_or_tensor, np.ndarray):
        #   image = image_or_path_or_tensor
        #   # pre_processed = True
        # elif type(image_or_path_or_tensor) == type(''):
        #   image = load_rec(image_or_path_or_tensor)
        # else:
        #   image = image_or_path_or_tensor['image'][0].numpy()
        #   pre_processed_images = image_or_path_or_tensor
        #   pre_processed = True 

        loaded_time = time.time()
        load_time += (loaded_time - start_time)
        detections = []
        # if not pre_processed:
        #   images = self.pre_process(image, meta)
        # else:
        #   images = pre_processed_images['images'][0]
        #   meta = pre_processed_images['meta'][scale]
        #   meta = {k: v.numpy()[0] for k, v in meta.items()}
        images = image_or_path_or_tensor
        images = images.to(self.opt.device)
        
        # torch.cuda().synchronize()
        pre_process_time = time.time()
        pre_time += 0

        output, dets, hm, forward_time = self.process(images, return_time=True)

        # torch.cuda.synchronize()
        net_time += forward_time - pre_process_time
        decode_time = time.time()
        dec_time += decode_time - forward_time

        if self.opt.debug >= 2:
            self.debug(debugger, images, dets, output)

        dets, name = self.post_process(dets, meta)
        if self.opt.gpus[0] >= 0: 
            torch.cuda.synchronize()
        post_process_time = time.time()
        post_time += post_process_time - decode_time 
        self.save_detection(hm, dets, self.opt.out_path, name = name)

        # torch.cuda.synchronize()
        end_time = time.time()
        tot_time += end_time - start_time
        # detections.append(dets)

        # results = self.merge_outputs(detections)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # merge_time += end_time - post_process_time
        # tot_time += end_time - start_time 

        # if self.opt.debug >= 1:
        #   self.show_results(debugger, image, results)
        return {'tot_time': tot_time, 'load': load_time, 'pre': pre_time, 'net': net_time, 'dec': dec_time}

        # return {'results': results, 'tot': tot_time, 'load': load_time,
        # 'pre': pre_time, 'net': net_time, 'dec': dec_time, 
        # 'post': post_time, 'merge': merge_time
        # }




