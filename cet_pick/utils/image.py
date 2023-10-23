from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random

def flip_ud(img):
	return np.flip(img, 1).copy()

def flip_lr(img):
	return np.flip(img, 2).copy()

def gaussian_radius(det_size, min_overlap=0.7):
	height, width = det_size

	a1  = 1
	b1  = (height + width)
	c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
	sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
	r1  = (b1 + sq1) / 2

	a2  = 4
	b2  = 2 * (height + width)
	c2  = (1 - min_overlap) * width * height
	sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
	r2  = (b2 + sq2) / 2

	a3  = 4 * min_overlap
	b3  = -2 * min_overlap * (height + width)
	c3  = (min_overlap - 1) * width * height
	sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
	r3  = (b3 + sq3) / 2
	return min(r1, r2, r3)

def gaussian3D_discrete(shape, sigma=1, label1=1, label2=2, thresh=0.5):
    m,n,o = [(ss - 1.) / 2. for ss in shape]
    y, x, z = np.ogrid[-m:m+1, -n:n+1, -o:o+1]
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # h[h>=0.5] = 1
    h[h >= thresh] = label1
    # h[h<0.5] = 2
    h[h < thresh] = label2
    return h

def gaussian3D(shape, sigma=1):
    m,n,o = [(ss - 1.) / 2. for ss in shape]
    y, x, z = np.ogrid[-m:m+1, -n:n+1, -o:o+1]
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # h[h>0.9] = 1
    return h

def draw_umich_gaussian_3d(heatmap, center, radius, label1, label2, thresh, k=1, discrete = True):
	diameter = 2 * radius + 1
	if discrete:
		gaussian = gaussian3D_discrete((diameter, diameter, diameter), sigma = diameter / 6, label1=label1, label2=label2, thresh=thresh)
	else:
		gaussian = gaussian3D((diameter, diameter, diameter), sigma = diameter / 6)
	x, y, z = int(center[0]), int(center[1]), int(center[2])
	depth, height, width = heatmap.shape[0:3]

	left, right = min(x, radius), min(width-x, radius+1)
	top, bottom = min(y, radius), min(height-y, radius+1)
	front, back = min(z, radius), min(depth-z, radius+1)

	masked_heatmap = heatmap[z - front:z + back, y - top:y + bottom, x - left: x + right]

	masked_gaussian = gaussian[radius-front:radius+back, radius-top:radius+bottom, radius-left:radius+right]

	if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
		np.maximum(masked_heatmap, masked_gaussian * k, out = masked_heatmap)
	return heatmap

def draw_msra_gaussian_3d(heatmap, center, sigma):
	tmp_size = sigma * 3
	mu_x = int(center[0] + 0.5)
	mu_y = int(center[1] + 0.5)
	mu_z = int(center[2] + 0.5)
	d, w, h= heatmap.shape[0], heatmap.shape[1], heatmap.shape[2]
	ulf = [int(mu_x - tmp_size), int(mu_y - tmp_size), int(mu_z - tmp_size)]
	brb = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1), int(mu_z+tmp_size+1)]

	if ulf[0] >= h or ulf[1] >= w or ulf[2] >= d or brb[0] < 0 or brb[1] < 0 or brb[2] < 0:
		return heatmap
	size = 2 * tmp_size + 1
	x = np.arange(0, size, 1, np.float32)
	y = x[:, np.newaxis]
	z = y[:,:,np.newaxis]
	x0 = y0 = z0 = size // 2
	g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))
	g_x = max(0, -ulf[0]), min(brb[0], h) - ulf[0]
	g_y = max(0, -ulf[1]), min(brb[1], w) - ulf[1]
	g_z = max(0, -ulf[2]), min(brb[2], d) - ulf[2]
	img_x = max(0, ulf[0]), min(brb[0], h)
	img_y = max(0, ulf[1]), min(brb[1], w)
	img_z = max(0, ulf[2]), min(brb[2], d)
	heatmap[img_z[0]:img_z[1],img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
	heatmap[img_z[0]:img_z[1],img_y[0]:img_y[1], img_x[0]:img_x[1]],
	g[g_z[0]:g_z[1], g_y[0]:g_y[1], g_x[0]:g_x[1]])
	return heatmap
