from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn.functional as F
import torch

def tomo_post_process(dets, z_dim_tot = 128):
	# dets batch, max_dets, dim 
	# return z coord based det dict 
	ret = []
	for i in range(dets.shape[0]):
		top_preds = {}
		z_dim = dets[i,:,2]
		for j in range(z_dim_tot):
			inds = (z_dim == j)
			if sum(inds) > 0:
				# print(sum(inds))
				top_preds[j] = dets[i, inds,:].astype(np.float32).tolist()
	ret.append(top_preds)
	return ret


def tomo_cluster_postprocess(cluster_center, feature_maps):

	b, dim, d, h, w = feature_maps.shape
	feature_maps = feature_maps.reshape((1, 16, -1))
	feature_maps = feature_maps.squeeze().T 
	cluster_center = cluster_center.unsqueeze(0)
	cluster_center = F.normalize(cluster_center)
	

	similarities = torch.matmul(feature_maps, cluster_center.T)
	similarities = similarities.reshape(d, h, w)

	similarities = F.sigmoid(similarities)
	similarities.unsqueeze(0)


	return similarities


