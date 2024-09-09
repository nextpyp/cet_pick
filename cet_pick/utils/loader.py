from __future__ import print_function, division

import os
import glob
import math
import numpy as np
from PIL import Image
import torch
import mrcfile
import cv2
import numpy as np
from numpy.lib import stride_tricks
from scipy.ndimage import gaussian_filter
import pandas as pd

def quantize(x, mi=-2.5, ma=2, dtype=np.uint8):
    if mi is None:
        mi = x.min()
    if ma is None:
        ma = x.max()
    r = ma - mi
    x = 255*(x - mi)/r
    x = np.clip(x, 0, 255)
    x = np.round(x).astype(dtype)
    return x

def load_rec(path, order='xyz', compress = False, is_tilt=False):

    with mrcfile.open(path, permissive=True) as mrc:
        rec = mrc.data
    if order == 'xzy' or order == 'xyz':
        if order == 'xzy':
            rec = np.swapaxes(rec, 2, 1)
        x, y, z = rec.shape
        new_z = math.ceil(z/2)

        if compress:
            new_slices = np.zeros([new_z, x, y])
        else:
            new_slices = np.zeros([z, x, y])
        j = 0
        if compress:
            for i in range(0, z, 2):
                new_slice = np.max(rec[:,:,i:i+2], axis = -1)
                # new_slice = rec[:,:,i]
                if is_tilt:
                    new_slice = (new_slice-new_slice.mean())/new_slice.std()
                new_slices[j,:,:] = new_slice
                j += 1
        else:
            for i in range(z):
                new_slice = rec[:,:,i]
                if is_tilt:

                    new_slice = (new_slice-new_slice.mean())/new_slice.std()
                new_slices[j,:,:] = new_slice
                j += 1
        if not is_tilt:
            new_slices = (new_slices - np.mean(new_slices))/np.std(new_slices)
    if order == 'zxy':
        z, x, y = rec.shape

        new_z = int(z//2)
        if compress:
            new_slices = np.zeros([new_z, x, y])
        else:
            new_slices = np.zeros([z, x, y])
        j = 0
        if compress:
            for i in range(0, z, 2):
                new_slice = np.max(rec[i:i+2,:,:], axis = 0)
                if is_tilt:
                    new_slice = (new_slice-new_slice.mean())/new_slice.std()
                new_slices[j,:,:] = new_slice
                j += 1
        else:
            for i in range(z):
                new_slice = rec[i]
                if is_tilt:
                    new_slice = (new_slice-new_slice.mean())/new_slice.std()
                new_slices[j,:,:] = new_slice
                j += 1
        if not is_tilt:
            new_slices = (new_slices - np.mean(new_slices))/np.std(new_slices)

    return new_slices

def preprocess(mrc, denoise=0, is_tilt=False):
    if denoise > 0:
        if is_tilt:
            dn_im = []
            for sli in mrc:
                dd = gaussian_filter(sli, sigma=denoise)
                dd = (dd-dd.mean())/dd.std()
                dd = quantize(dd)
                dd = cv2.normalize(dd, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                dn_im.append(dd)
            im = np.asarray(dn_im)
        else:
            im = gaussian_filter(mrc, sigma=denoise)
            im = (im-im.mean())/im.std()
            im = quantize(im,mi=-3,ma=3)
            im = (im - np.min(im))/(np.max(im) - np.min(im))
    else:
        if is_tilt:
            dn_im = []
            for sli in mrc:
                # dd = gaussian_filter(sli, sigma=sigma)
                dd = sli
                dd = (dd-dd.mean())/dd.std()
                dd = quantize(dd)
                dd = cv2.normalize(dd, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                dn_im.append(dd)
            im = np.asarray(dn_im)
        else:
            im = (mrc-mrc.mean())/mrc.std()
            im = quantize(im)
            im = (im - np.min(im))/(np.max(im) - np.min(im))
    return im 


def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6

def load_tlt(path):
    angles = pd.read_table(path, header=None)
    angles = angles.to_numpy()
    return angles

def load_tomo_all_and_angles_from_list(names, tilt_paths, rec_paths, angle_paths, order_tilt = 'zxy', order_rec= 'xzy',compress=False, denoise=0):
    tilt_ims = {}
    rec_ims = {}
    angles = {}
    for name, tilt_path, rec_path, angle_path in zip(names, tilt_paths, rec_paths, angle_paths):
        tilt_im = load_rec(tilt_path, order=order_tilt,compress = False, is_tilt=True)
        rec_im = load_rec(rec_path, order = order_rec, compress= compress, is_tilt=False)
        tilt_angle = load_tlt(angle_path)
        tilt_im = preprocess(tilt_im, denoise=denoise, is_tilt=True)
        rec_im = preprocess(rec_im, denoise=denoise, is_tilt=False)
        tilt_ims[name] = tilt_im 
        rec_ims[name] = rec_im 
        angles[name] = tilt_angle
    return tilt_ims, rec_ims, angles

def load_tomos_and_angles_from_list(names, tomo_paths, angle_paths, order='xzy', compress=False, denoise=0, tilt=False):
    images = {}
    angles = {}
    for name, tomo_path, angle_path in zip(names, tomo_paths, angle_paths):
        im = load_rec(tomo_path, order=order, compress=compress, is_tilt = tilt)
        tilt_angle = load_tlt(angle_path)
        im = preprocess(im, denoise=denoise, is_tilt=tilt)
        images[name] = im
        angles[name] = tilt_angle
    return images, angles

def load_tomos_from_list(names, paths, order='xzy', compress=False, denoise=0, tilt=False):
    images = {}
    for name, path in zip(names, paths):
        im = load_rec(path, order=order, compress=compress, is_tilt = tilt)
        # im = preprocess(im, denoise=denoise, is_tilt = tilt, sigma=1)
        im = preprocess(im, denoise=denoise, is_tilt = tilt)

        images[name] = im
    return images

def load_tomos_from_list_nopre(names, paths, order='xzy', compress=False, denoise=False, tilt=False):
    images = {}
    for name, path in zip(names, paths):
        im = load_rec(path, order=order, compress=compress, is_tilt = tilt)
        images[name] = im
    return images