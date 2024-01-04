from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import pickle
from torch import Tensor
from typing import Tuple

def clip_img(img: Tensor, inplace: bool = False) -> Tensor:
    """Clip tensor data so that it is within a valid image range. That is
    between 0-1 for float images and 0-255 for int images. Values are clamped
    meaning any values outside this range are set to the limits, other values
    are not touched.

    Args:
        img (Tensor): Image or batch of images to clip.
        inplace (bool, optional): Whether to do the operation in place.
            Defaults to False; this will first clone the data.

    Returns:
        Tensor: Reference to input image or new image.
    """
    if not inplace:
        img = img.clone()
    if img.is_floating_point():
        c_min, c_max = (0, 1)
    else:
        c_min, c_max = (0, 255)
    return torch.clamp_(img, c_min, c_max)


class Crop2d(nn.Module):
    """Crop input using slicing. Assumes BCHW data.

    Args:
        crop (Tuple[int, int, int, int]): Amounts to crop from each side of the image.
            Tuple is treated as [left, right, top, bottom]/
    """

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x: Tensor) -> Tensor:
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


class Shift2d(nn.Module):
    """Shift an image in either or both of the vertical and horizontal axis by first
    zero padding on the opposite side that the image is shifting towards before
    cropping the side being shifted towards.

    Args:
        shift (Tuple[int, int]): Tuple of vertical and horizontal shift. Positive values
            shift towards right and bottom, negative values shift towards left and top.
    """

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_block(x)
        

def _center_distance(centers1, centers2):
	# center1 is output from tomo decode with B * Num * 5 shape
	# center 2 is ground truth with B * Num * 3
	preds_coord = centers1[:,:, :3]
	dist_mat = torch.cdist(preds_coord, centers2)
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


_verbose = False

def log(msg):
    print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
    sys.stdout.flush()

def vlog(msg):
    if _verbose:
        print('{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg))
        sys.stdout.flush()

def flog(msg, outfile):
    msg = '{}     {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(msg)
    sys.stdout.flush()
    try:
        with open(outfile,'a') as f:
            f.write(msg+'\n')
    except Exception as e:
        log(e)

def load_pkl(pkl):
    with open(pkl,'rb') as f:
        x = pickle.load(f)
    return x

def save_pkl(data, out_pkl, append='False'):
    mode = 'wb' if append == False else 'ab'
    if mode == 'wb' and os.path.exists(out_pkl):
        vlog('Warning: {out_pkl} already exists. Overwriting.')
    with open(out_pkl, mode) as f:
        pickle.dump(data, f)

def R_from_eman(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    # handling EMAN convention mismatch for where the origin of an image is (bottom right vs top right)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1
    return R

def R_from_relion(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1
    return R

def xrot(tilt_deg):
    '''Return rotation matrix associated with rotation over the x-axis'''
    theta = tilt_deg*np.pi/180
    tilt = np.array([[1.,0.,0.],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return tilt
    
def zero_sphere(vol):
    '''Zero values of @vol outside the sphere'''
    assert len(set(vol.shape)) == 1, 'volume must be a cube'
    D = vol.shape[0]
    xx = np.linspace(-1, 1, D, endpoint=True if D % 2 == 1 else False)
    z,y,x = np.meshgrid(xx,xx,xx)
    coords = np.stack((x,y,z),-1)
    r = np.sum(coords**2,axis=-1)**.5
    vlog('Zeroing {} pixels'.format(len(np.where(r>1)[0])))
    vol[np.where(r>1)] = 0
    return vol

	
