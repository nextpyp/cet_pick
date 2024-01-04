from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from cet_pick.datasets.dataset_factory import dataset_factory
from cet_pick.detectors.detector_factory import detector_factory
from cet_pick.utils.loader import load_rec, preprocess

class PrefetchDataset(torch.utils.data.Dataset):

    def __init__(self, opt, dataset):
        self.opt = opt 
        self.images = dataset.images 
        # self.images = dataset.paths
        # self.images = dataset.tomos
        # self.targets = dataset.targets 
        # self.targets = dataset.gt_dets
        self.names = dataset.names
    def _downscale_coord(self,ann):
        x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]
        return [x,y,z]

    def __getitem__(self, index): 
        img = self.images[index]
        # coords = self.targets[index]
        name = self.names[index]
        # inp = load_rec(img, order='xzy', compress=True, is_tilt=False)
        # inp = preprocess(inp, denoise=True, is_tilt=False, sigma=1)
        # print('inp', inp.shape)
        # inp = inp[59:184]
        # print('inp', inp.mean())
        # print('inp std', inp.std())
        # gt_det = []
        # num_objs = len(coords)
        # for k in range(num_objs):
        #     ann = coords[k]
        #     ann = self._downscale_coord(ann)
        #     # print('ann', ann)
        #     ann = np.array(ann)
        #     gt_det.append(ann)
        # gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
        ret = {'input':img.astype(np.float32)}
        meta = {'name': name,'zdim':img.shape[0]}
        ret['meta'] = meta
        # print('ret inp', ret['input'].mean())
        # print('ret inp', ret['input'].std())
        return ret 

    def __len__(self):
        return len(self.images)

        


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Dataset = dataset_factory[opt.task]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]

    detector = Detector(opt)
    # split = 'val' if not opt.trainval else 'test'
    split = 'test'
    dataset = Dataset(opt, split)
    data_loader = torch.utils.data.DataLoader(PrefetchDataset(opt, dataset), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot_time', 'load', 'pre', 'net', 'dec']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for iter_id, batch in enumerate(data_loader):
        input_tomo = batch['input']
        meta = batch['meta']
        ret = detector.run(input_tomo, meta)
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   iter_id, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
        avg_time_stats[t].update(ret[t])
        Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
    bar.finish()

if __name__ == '__main__':
    opt = opts().parse()
    test(opt)

