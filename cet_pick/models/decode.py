from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from cet_pick.models.utils import _gather_feat, _transpose_and_gather_feat
def _nms(heat, kernel=5):
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

def _topk(scores, K = 900):
    batch, channel, depth, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, channel, -1), K)

    topk_zs, topk_ys, topk_xs = _convert_1d_to_3d(topk_inds, depth, height, width)
    topk_inds = topk_inds.view(batch, K)
    topk_zs = topk_zs.view(batch, K)
    topk_ys = topk_ys.view(batch, K)
    topk_xs = topk_xs.view(batch, K)

    return topk_scores, topk_zs, topk_ys, topk_xs, topk_inds


def tomo_decode(heat, reg=None, K = 900):
	batch, cat, depth, height, width = heat.size()
	heat = _nms(heat)
	scores, zs, ys, xs, inds = _topk(heat, K=K)
	if reg is not None:
		# print('not none')
		reg = _transpose_and_gather_feat(reg, inds)
		reg = reg.view(batch, K, 2)
		xs = xs.view(batch, K, 1) + reg[:,:,0:1].float()
		ys = ys.view(batch, K, 1) + reg[:,:,1:2].float()
		zs = zs.view(batch, K, 1)
	else:
		# print('is none')
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
	


