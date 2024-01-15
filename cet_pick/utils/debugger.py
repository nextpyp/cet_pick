
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

class Debugger(object):
    def __init__(self, num_class = 1, dataset=None, down_ratio=4):
        import matplotlib.pyplot as plt
        self.plt = plt 
        self.imgs = {}
        self.dim_scale = 1
        colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        if dataset == 'tomo' or 'cr' or 'denoise':
            self.names = ['particle']
            self.num_class = 1 
        num_classes = len(self.names)

        self.down_ratio = down_ratio

    def add_slice(self, tomos, img_id = ''):
        # extract slice of tomo images 
        self.imgs[img_id] = tomos.copy()

    def add_mask(self, mask, bg, slice_num = 0, trans = 0.8):
        self.imgs[slice_num] = (mask.reshape(mask.shape[0], mask.shape[1], 1) * 255 * trans + bg * (1 - trans)).astype(np.uint8)

    def show_img(self, pause = False, slice_num = 0):
        cv2.imshow('{}'.format(slice_num), self.imgs[slice_num])
        if pause:
            cv2.waitKey()

    def gen_colormap(self, img, output_res = None):
        img = img.copy()
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)

        color_map = (img * colors).max(axis=2).astype(np.uint8)

        color_map = cv2.resize(color_map, (output_res[1], output_res[0]))
        return color_map

    def add_blend_img(self, back, fore, img_id = 'blend', trans=0.7):
        # fore = 255 - fore
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = (back * (1. - trans) + fore * trans)
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

    def add_circle(self, center, wh, c, slice_num = 0):
        cv2.circle(self.imgs[slice_num], (center[0], center[1]), wh, c, 2)

    def show_all_imgs(self, pause=False, time = 0):
        for i, v in self.imgs.items():
            cv2.imshow('{}'.format(i), v)
        if cv2.waitKey(0 if pause else 1) == 27:
            import sys 
            sys.exit(0)

    def save_img(self, slice_num = 0, path = './cache/debug/'):
        cv2.imwrite(path + '{}.png'.format(slice_num), self.imgs[slice_num])

    def save_detection(self, dets, path='./cache/debug/', prefix='', name=''):

        out_detect = open(path+'/{}_{}.txt'.format(name, prefix), 'w+')
        for k, v in dets.items():
            for c in v:
                x, y, z = int(c[0]), int(np.floor(c[1])), int(np.floor(c[2]))
                score = c[3]
                conf = c[4]
                if conf > 0.3:
                    print(str(x) + '\t' + str(z) + '\t' + str(y) + '\t' + str(score), file = out_detect)






    def save_all_imgs(self, path='./cache/debug/', prefix='', slice_num = 0, genID=False):
        if genID:
            try:
                idx = int(np.loadtxt(path + '/id.txt'))
            except:
                idx = 0
            prefix=idx 
            np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
        for i, v in self.imgs.items():
            cv2.imwrite(path + '/{}{}{}.png'.format(prefix, i, slice_num), v)

    def add_particle_detection(self, dets, rad, show_box=False, center_thresh = 0.3, img_id = 'det'):
        for i in range(len(dets)):
            # print(dets[i])
            if len(dets[0]) > 3:
                if dets[i][4] > center_thresh:
                    # print('dets', dets[i])
                    ct = dets[i][:2] * self.down_ratio
                    # rad = dets[i, -2].astype(np.int32)
                    cv2.circle(self.imgs[img_id], (int(ct[0]), int(ct[1])), rad, (255,0,0), 1)
            else:
                ct = dets[i][:2] * self.down_ratio
                
                cv2.circle(self.imgs[img_id], (int(ct[0]), int(ct[1])), rad, (255,0,0), 1)

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255





