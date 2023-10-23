from __future__ import print_function, division

import os
import glob

import numpy as np
from PIL import Image
import torch
import mrcfile
import cv2
import numpy as np

def quantize(x, mi=-3, ma=3, dtype=np.uint8):
    if mi is None:
        mi = x.min()
    if ma is None:
        ma = x.max()
    r = ma - mi
    x = 255*(x - mi)/r
    x = np.clip(x, 0, 255)
    x = np.round(x).astype(dtype)
    return x


def load_rec(path):
    # print('path', path)
    with mrcfile.open(path, permissive=True) as mrc:
        rec = mrc.data 
    # check if it has NaN values
    if(np.isnan(rec).any()):
        # print("The Array contain NaN values")
        raise ValueError("Input contains NaN values")
    # else:
    #     print("The Array does not contain NaN values")
    rec = np.swapaxes(rec, 2, 1)
    x, y, z = rec.shape 
    new_z = int(z//2)
    # new_z = z
    new_slices = np.zeros([new_z, x, y])
    j = 0
    for i in range(0, z, 2):
        new_slice = np.max(rec[:,:,i:i+2], axis = -1)
        # new_slice = rec[:,:,i]
        new_slice = (new_slice-new_slice.mean())/new_slice.std()
        new_slice = quantize(new_slice)
        new_slice = cv2.normalize(new_slice, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        new_slices[j,:,:] = new_slice
        j += 1
        # print(j)
        # print(np.std(new_slice))
    new_slices = (new_slices - np.min(new_slices))/(np.max(new_slices) - np.min(new_slices))
    new_slices = (new_slices - np.mean(new_slices))/np.std(new_slices)
    return new_slices

# def load_rec(path):
# #     # print('path', path)
#     with mrcfile.open(path, permissive=True) as mrc:
#         rec = mrc.data 
#     rec = np.swapaxes(rec, 2, 1)
#     x, y, z = rec.shape 
#     new_z = int(z)
#     new_slices = np.zeros([new_z, x, y])
#     j = 0
#     for i in range(0, z):
#         # new_slice = np.max(rec[:,:,i:i+2], axis = -1)
#         new_slice = rec[:,:,i]
#         new_slice = (new_slice-new_slice.mean())/new_slice.std()
#         new_slice = quantize(new_slice)
#         new_slice = cv2.normalize(new_slice, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         new_slices[j,:,:] = new_slice
#         j += 1
#     new_slices = (new_slices - np.min(new_slices))/(np.max(new_slices) - np.min(new_slices))
#     new_slices = (new_slices - np.mean(new_slices))/np.std(new_slices)
#     return new_slices


def load_tomos_from_list(names, paths):
    images = {}
    for name, path in zip(names, paths):
        # print('path', path)
        im = load_rec(path)
        images[name] = im 
    return images
