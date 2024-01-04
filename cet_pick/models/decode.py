from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np 
from cet_pick.models.utils import _gather_feat, _transpose_and_gather_feat


def _nms_xy(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(
        heat, (1, kernel, kernel), stride=1, padding=(0,pad,pad))
    keep = (hmax == heat).float()
    return heat * keep

def _nms_z(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(
        heat, (kernel, 1, 1), stride=1, padding=(pad,0,0))
    keep = (hmax == heat).float()
    return heat * keep

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(
        heat, (3, kernel, kernel), stride=1, padding=(1,pad,pad))
    keep = (hmax == heat).float()
    return heat * keep

def _convert_1d_to_3d(inds, d, h, w):
    z_coord = torch.floor(inds.float()/(h*w)).int()
    t = inds.int() - (z_coord * h * w)
    y_coord = torch.floor(t.float() / w)
    x_coord = t % h
    
    return z_coord, y_coord, x_coord
def non_maximum_suppression_3d(x, d, scale=1.0, threshold=-np.inf):
    ## enumerate coordinate deltas within d
    r = scale*d/2
    width = int(np.ceil(r))
    A = np.arange(-width,width+1)
    ii,jj,kk = np.meshgrid(A, A, A)
    mask = (ii**2 + jj**2 + kk**2) <= r*r
    ii = ii[mask]
    jj = jj[mask]
    kk = kk[mask]
    zstride = x.shape[1]*x.shape[2]
    ystride = x.shape[2]
    coord_deltas = ii*zstride + jj*ystride + kk
    
    A = x.ravel()
    I = np.argsort(A, axis=None)[::-1] # reverse to sort in descending order
    S = set() # the set of suppressed coordinates

    scores = np.zeros(len(A), dtype=np.float32)
    coords = np.zeros((len(A),3), dtype=np.int32)

    j = 0
    for i in I:
        if A[i] <= threshold:
            break
        if i not in S:
            ## coordinate i is next center
            zz,yy,xx = np.unravel_index(i, x.shape)
            scores[j] = A[i]
            coords[j,0] = xx
            coords[j,1] = yy
            coords[j,2] = zz
            j += 1
            ## add coordinates within d of i to the suppressed set
            for delta in coord_deltas:
                S.add(i + delta)
    
    return scores[:j], coords[:j]


def _topk(scores, K = 900):
    batch, channel, depth, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, channel, -1), K)

    topk_zs, topk_ys, topk_xs = _convert_1d_to_3d(topk_inds, depth, height, width)
    topk_inds = topk_inds.view(batch, K)
    topk_zs = topk_zs.view(batch, K)
    topk_ys = topk_ys.view(batch, K)
    topk_xs = topk_xs.view(batch, K)

    return topk_scores, topk_zs, topk_ys, topk_xs, topk_inds

# def tomo_decode_fiber(heat, kernel = 3, reg=None, K=900):
#     batch, cat, depth, height, width = heat.size()
#     heat_pre = _nms_xy(heat, kernel=kernel)
#     heat_pre = _nms_z(heat_pre, kernel=kernel)
#     # device = heat.get_device()
#     heat_pre = heat_pre.squeeze().cpu().numpy()
#     # scores, coords = non_maximum_suppression_3d(heat_pre, 5, 0.2)
#     scores, coords = scores[:K], coords[:K]
#     coords = torch.as_tensor(coords).unsqueeze(0).to(heat.device)
#     scores = torch.as_tensor(scores).unsqueeze(0).unsqueeze(-1).to(heat.device)
#     detections = torch.cat([coords, scores,scores], dim=2)
#     # print('-----fiber detect------', detections.shape)
#     return detections

def tomo_decode(heat, kernel = 3, reg=None, K = 900, if_fiber = False):
    batch, cat, depth, height, width = heat.size()

    if if_fiber:
        heat = _nms_xy(heat, kernel=kernel)
        heat = _nms_z(heat, kernel=kernel)
    else:
        heat = _nms(heat, kernel=kernel)
    scores, zs, ys, xs, inds = _topk(heat, K=K)
    if reg is not None:

        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:,:,0:1].float()
        ys = ys.view(batch, K, 1) + reg[:,:,1:2].float()
        zs = zs.view(batch, K, 1)
    else:

        xs = xs.view(batch, K, 1) + 0.25
        ys = ys.view(batch, K, 1) + 0.25
        zs = zs.view(batch, K, 1)
    # wh = _transpose_and_gather_feat(wh, inds)
    # wh = wh.view(batch, K, 1).float()
    # wh = torch.zeros(batch, K, 1).float()
    scores = scores.view(batch, K, 1).float()
    xs = xs.float()
    ys = ys.float()
    zs = zs.float()


    centers = torch.cat([xs, ys, zs], dim = 2)
    detections = torch.cat([centers, scores, scores], dim = 2)
    return detections


def get_roi_regions(feat, rois, depth, radius):
    pass
    


