from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _center_distance(centers1, centers2):
	# center1 is output from tomo decode with B * Num * 5 shape
	# center 2 is ground truth with B * Num * 3
	preds_coord = centers1[:,:, :3]
	dist_mat = torch.cdist(preds_coord, centers2)
	# print(dist_mat.shape)
	return dist_mat


def _sigmoid(x):
	y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
	return y

def _gather_feat(feat, ind, mask=None):
	# input is N, D*W*H, C
	dim = feat.size(2)
	# [N, max] -> [N, max, 1] -> [N, max, C]
	ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
	feat = feat.gather(1, ind)
	if mask is not None:
		mask = mask.unsqueeze(2).expand_as(feat)
		feat = feat[mask]
		feat = feat.view(-1, dim)
	return feat 




def _transpose_and_gather_feat(feat, ind):
	# transform from N, C, D, W, H to N, D, W, H, C
	feat = feat.permute(0,2,3,4,1).contiguous()
	# N, D, W, H, C to N, D*W*H, C
	feat = feat.view(feat.size(0), -1, feat.size(4))
	feat = _gather_feat(feat, ind)

	return feat

def insize_from_outsize_3d(layers, outsize_z, outsize_xy):
    """ calculates in input size of a convolution stack given the layers and output size """
    for layer in layers[::-1]:
        if hasattr(layer, 'kernel_size'):
            kernel_size = layer.kernel_size
            print('kernel_size', kernel_size)
            if type(kernel_size) is tuple:
                kernel_size_z = kernel_size[0]
                kernel_size_xy = kernel_size[1] # assume square
            else:
                kernel_size_z = 1 
                kernel_size_xy = kernel_size
            # if type(kernel_size) is tuple:
            #     kernel_size = kernel_size[0]
        else:
            kernel_size_z, kernel_size_xy = 1, 1
        if hasattr(layer, 'stride'):
            stride = layer.stride
            
            if type(stride) is tuple:
                stride_z = stride[0]
                stride_xy = stride[1]
            else:
                stride_z = 1 
                stride_xy = stride
        else:
            stride_z, stride_xy = 1, 1
        if hasattr(layer, 'padding'):
            pad = layer.padding
            if type(pad) is tuple:
                pad_z = pad[0]
                pad_xy = pad[1]
            else:
                pad_z = 0 
                pad_xy = pad
        else:
            pad_z, pad_xy = 0, 0

        if hasattr(layer, 'dilation'):
            dilation = layer.dilation
            # print('layer dilation',dilation)
            if type(dilation) is tuple:
                dilation_z = dilation[0]
                dilation_xy = dilation[1]
            else:
                dilation_z = 1 
                dilation_xy = dilation
        else:
            # print('does else work')
            dilation_z, dilation_xy = 1, 1
        # print('dilation', dilation)
        outsize_z = (outsize_z-1)*stride_z+1+(kernel_size_z-1)*dilation_z-2*pad_z
        outsize_xy = (outsize_xy-1)*stride_xy + 1 + (kernel_size_xy-1)*dilation_xy - 2*pad_xy 
        # print('layer', layer.__dict__)
        print('outsize_z', outsize_z)
        print('outsize_xy',outsize_xy)
    return outsize_z, outsize_xy

def out_from_in(layer, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    if hasattr(layer, 'kernel_size'):
        kernel_size = layer.kernel_size # assume square
        if type(kernel_size) is tuple:
            kernel_size = kernel_size[0]
    if hasattr(layer, 'stride'):
        stride = layer.stride
        if type(stride) is tuple:
            stride = stride[0]
    else:
        stride = 1
    if hasattr(layer, 'padding'):
        pad = layer.padding
        if type(pad) is tuple:
            pad = pad[0]
    else:
        pad = 0
    if hasattr(layer, 'dilation'):
        dilation = layer.dilation
        if type(dilation) is tuple:
            dilation = dilation[0]
    else:
        dilation = 1
    n_out = math.floor((n_in - dilation*(kernel_size-1))-1+2*pad)/stride + 1 
    actualP = (n_out-1)*stride-n_in+kernel_size
    pR = math.ceil(actualP/2)
    pL = math.ceil(actualP/2)
    j_out = j_in * stride 
    r_out = r_in + (dilation*(kernel_size-1))*j_in 
    start_out = start_in + (dilation*(kernel_size-1)/2 - pL)*j_in
    return n_out, j_out, r_out, start_out


	
