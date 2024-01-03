import numpy as np 
import mrcfile
from sklearn.cluster import KMeans
import matplotlib
from sklearn.manifold import TSNE
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os 
import argparse 
import os 
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
import json

def quantize(x, mi=-2.5, ma=3, dtype=np.uint8):
    if mi is None:
        mi = x.min()
    if ma is None:
        ma = x.max()
    r = ma - mi
    x = 255*(x - mi)/r
    x = np.clip(x, 0, 255)
    x = np.round(x).astype(dtype)
    return x

def load_rec(path, order='xyz', compress = False):
    with mrcfile.open(path, permissive=True) as mrc:
        rec = mrc.data 
    if order == 'xzy' or order == 'xyz':
        if order == 'xzy':
            rec = np.swapaxes(rec, 2, 1)
        x, y, z = rec.shape 
        new_z = int(z//2)
        if compress:
            new_slices = np.zeros([new_z, x, y])
        else:
            new_slices = np.zeros([z, x, y])
        j = 0
        if compress:
            for i in range(0, z, 2):
                new_slice = np.max(rec[:,:,i:i+2], axis = -1)
                # new_slice = rec[:,:,i]
                new_slice = (new_slice-new_slice.mean())/new_slice.std()
                new_slice = quantize(new_slice)
                new_slice = cv2.normalize(new_slice, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                new_slices[j,:,:] = new_slice
                j += 1
        else:
            for i in range(z):
                new_slice = rec[:,:,i]
                new_slice = (new_slice-new_slice.mean())/new_slice.std()
                new_slice = quantize(new_slice)
                new_slice = cv2.normalize(new_slice, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                new_slices[j,:,:] = new_slice
                j += 1

        new_slices = (new_slices - np.min(new_slices))/(np.max(new_slices) - np.min(new_slices))
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
                # new_slice = rec[:,:,i]
                new_slice = (new_slice-new_slice.mean())/new_slice.std()
                new_slice = quantize(new_slice)
                new_slice = cv2.normalize(new_slice, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                new_slices[j,:,:] = new_slice
                j += 1
        else:
            for i in range(z):
                new_slice = rec[i]
                new_slice = (new_slice-new_slice.mean())/new_slice.std()
                new_slice = quantize(new_slice)
                new_slice = cv2.normalize(new_slice, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                new_slices[j,:,:] = new_slice
                j += 1
        
        new_slices = (new_slices - np.min(new_slices))/(np.max(new_slices) - np.min(new_slices))
        new_slices = (new_slices - np.mean(new_slices))/np.std(new_slices)
    return new_slices


def add_arguments(parser):
    parser.add_argument('--input',type=str, help='input npz file with names and coords info')
    parser.add_argument('--color', type=str, help='color np file that contains all painted color')
    parser.add_argument('--dir_simsiam', type=str, help='directory path for simsiam')
    parser.add_argument('--rec_dir', type=str, help='path to directory with corresponding rec file')
    parser.add_argument('--compress', action='store_true', 
                              help = 'whether to combine 2 slice into 1 slice during reading of the dataset')
    return parser 


def get_3d_hm(volume, coords, labels, names, use_name, out_dir):
    
    z, r1, c1= volume.shape
    
    all_inds = np.arange(labels.shape[0])

    coords_use = coords[np.where(names == use_name)]
    labels_used = labels[np.where(names == use_name)]
    coords = coords_use
    labels = labels_used
    assert coords.shape[0] == labels.shape[0]
    unique_slices = np.unique(coords[:,-1])
    num_of_slices = unique_slices.shape[0]
    hm_all = np.zeros((z, r1, c1, 3), dtype=np.uint8)
    rec_all = np.zeros((z, r1, c1, 3), dtype=np.uint8)
    for i in range(z):
        rec_single  = volume[i]
        rec_single -= rec_single.mean()
        rec_single /= rec_single.std()
        rec_single = quantize(rec_single)
        rec_3d = np.dstack((rec_single, rec_single, rec_single))
        rec_all[i] = rec_3d
    rec_all = gaussian_filter(rec_all, sigma=0.8)
    rec_out_path = os.path.join(out_dir, use_name) + '_rec3d.npy'

    np.save(rec_out_path, rec_all)
    for i in range(num_of_slices):
        hm = np.zeros((int(r1), int(c1), 3), np.uint8)
        slice_num = unique_slices[i]
        coord_slice = coords[np.where(np.logical_and(coords[:,-1]>=slice_num - 2,coords[:,-1]<=slice_num+2))]
        label_slice = labels[np.where(np.logical_and(coords[:,-1]>=slice_num - 2,coords[:,-1]<=slice_num+2))]
        for ind, c in enumerate(coord_slice):
            color = label_slice[ind]
            cv2.circle(hm, (int(c[0]), int(c[1])), int(12-(abs(c[-1]-slice_num))), (int(color[0]), int(color[1]), int(color[2])), -1)
        hm_all[int(slice_num)] = hm
    hm_out_path = os.path.join(out_dir, use_name) + '_hm3d_simsiam.npy'
    np.save(hm_out_path, hm_all)

def main(args):
    data = np.load(args.input)
    names = data['name']
    coords = data['coords']
    unique_names = np.unique(names)
    colors = np.load(args.color)

    for nm in unique_names:
        print('constructing 3D tomogram visualization for {}....'.format(nm))
        rec = os.path.join(args.rec_dir, nm) +'.rec'
        rec_single = load_rec(rec, order='xzy', compress=args.compress)
        get_3d_hm(rec_single, coords, colors, names, nm, args.dir_simsiam)


if __name__=='__main__':
    parser = argparse.ArgumentParser('Script for visualizing 3D tomogram visualization')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
    