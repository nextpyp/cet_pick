from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn.functional as F
import torch
from sknetwork.topology import get_connected_components, get_largest_connected_component
from scipy import sparse

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
                top_preds[j] = dets[i, inds,:].astype(np.float32).tolist()
    ret.append(top_preds)
    return ret

def k_x(y, a,b,c):
    k = (2*a)/((1+(2*a*y+b)**2))**(2/3)
    return np.max(k)

def tomo_group_postprocess(dets_all, distance_cutoff=15, min_per_group = 5):
    output_coords = []
    dets_w_score = np.asarray(dets_all)
    dets = dets_w_score[:,:3]
    adj_matrix = np.zeros((dets.shape[0], dets.shape[0]))
    for ind in range(dets.shape[0]):
        dist_with_curr = np.sqrt(np.sum((dets[ind] - dets) ** 2, 1))
        connect_ind = np.where(dist_with_curr <= distance_cutoff)[0]
        adj_matrix[ind, connect_ind] = 1
    adjacency = sparse.csr_matrix(adj_matrix)
    labels = get_connected_components(adjacency)
    labels_unique = np.unique(labels)
    # potential_all = []
    for lb in labels_unique:
        potential_candidates = dets_w_score[np.where(labels == lb)[0]]
        if potential_candidates.shape[0] > min_per_group:
            # potential_all.append(potential_candidates)
            for jj in range(potential_candidates.shape[0]):
                output_coords.append(potential_candidates[jj])
    return output_coords
    
def tomo_fiber_postprocess(dets, distance_cutoff=15, res_cutoff = 30, curvature_cutoff=0.03):
    output_coords = []
    dets = np.asarray(dets)
    adj_matrix = np.zeros((dets.shape[0], dets.shape[0]))
    for ind in range(dets.shape[0]):
        dist_with_curr = np.sqrt(np.sum((dets[ind] - dets) ** 2, 1))
        connect_ind = np.where(dist_with_curr <= distance_cutoff)[0]
        adj_matrix[ind, connect_ind] = 1
    adjacency = sparse.csr_matrix(adj_matrix)
    labels = get_connected_components(adjacency)
    labels_unique = np.unique(labels)
    potential_all = []
    for lb in labels_unique:
        potential_candidates = dets[np.where(labels == lb)[0]]
        if potential_candidates.shape[0] > 6:
            potential_all.append(potential_candidates)
    for i in range(len(potential_all)):
        curr_line = potential_all[i].copy()
        curr_line[:, [1, 0]] = curr_line[:, [0, 1]]
        diff_maxmin = np.max(curr_line[:,1]) - np.min(curr_line[:,1])
        num_points = diff_maxmin // 2
        y_range = np.linspace(np.min(curr_line[:,1])-1, np.max(curr_line[:,1])+1, int(num_points)) ## prevent falling on same point
        if y_range.shape[0] > 0:
            p_yx = np.polyfit(curr_line[:,1],curr_line[:,0],2, full=True)
            num_points_fit = curr_line.shape[0]
            p_yz = np.polyfit(curr_line[:,1],curr_line[:,2],2, full=True);
            coeffs_yx = p_yx[0]
            if p_yx[1].shape[0] > 0:
                res_x = p_yx[1][0]/num_points_fit
            else:
                res_x = 10000
            coeffs_yz = p_yz[0]
            if p_yz[1].shape[0] > 0:
                res_z = p_yz[1][0]/num_points_fit
            else:
                res_z = 10000
            kx = k_x(y_range, *coeffs_yx)
            kz = k_x(y_range, *coeffs_yz)
            if res_x + res_z < res_cutoff: 
                if abs(kx) < curvature_cutoff and abs(kz) < curvature_cutoff:
                    x_out = np.polyval(coeffs_yx, y_range)
                    z_out = np.polyval(coeffs_yz, y_range)
                    for jj in range(x_out.shape[0]):
                        line = [int(y_range[jj]), int(z_out[jj]), int(x_out[jj])]
                        output_coords.append(line)
            elif res_x + res_z < res_cutoff * 3:
                if abs(kx) < curvature_cutoff/10 and abs(kz) < curvature_cutoff/10:
                    x_out = np.polyval(coeffs_yx, y_range)
                    z_out = np.polyval(coeffs_yz, y_range)
                    for jj in range(x_out.shape[0]):
                        line = [int(y_range[jj]), int(z_out[jj]), int(x_out[jj])]
                        output_coords.append(line)
    return output_coords

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


