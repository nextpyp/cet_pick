from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from typing import Tuple, List, Optional
import numbers
import cv2
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from torch import Tensor
import math
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale

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

    # hmax = nn.functional.max_pool3d(
    #     heat, (1, kernel, kernel), stride=1, padding=(0,pad,pad))
    hmax = nn.functional.max_pool3d(
        heat, (kernel, kernel, kernel), stride=1, padding=(pad,pad,pad))
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

def _top_scores(scores, cutoff = 0.5):
    batch, channel, depth, height, width = scores.size()
    _, _, top_zs, top_ys, top_xs = torch.where(scores > cutoff)
    num_of_coords = top_zs.shape[0]
    top_zs = top_zs.view(batch, num_of_coords)
    top_ys = top_ys.view(batch, num_of_coords)
    top_xs = top_xs.view(batch, num_of_coords)
    return top_zs, top_ys, top_xs



def get_potential_coords_pyramid(rec, sigmas = [2,4], num_pyramid = 3, kernel = 3):
    ims = []
    z, r, c = rec.shape
    bound_x, bound_y = 30, 30 
    if r > 512 and c > 512:
        bound_x, bound_y = bound_x * 2, bound_y * 2
    #     # do 2x downsampling first 
    #     down_rec = []
    #     for i in range(z):
    #         down_rec.append(rescale(rec[i],0.5,anti_aliasing=True))
    #     down_rec = np.asarray(down_rec)
    # rec = down_rec
    # sigma = sigma_init
    num_pyramid = len(sigmas)
    for i in range(num_pyramid):
        # sigma = sigma*(i+1)
        sigma = sigmas[i]
        im = gaussian_filter(rec, sigma)
        ims.append(im)
        # sigma *= 2
    diff_all = []
    for i in range(num_pyramid-1):
        diff = ims[i+1] - ims[i]
        diff[:10,:,:] = 0
        diff[-10:,:,:]= 0
        diff[:,:bound_x,:] = 0 
        diff[:,-bound_x:,:] = 0
        diff[:,:,:bound_y] = 0
        diff[:,:,-bound_y:] = 0
        diff = torch.as_tensor(diff)
        diff = diff.unsqueeze(0).unsqueeze(0) 
        nms_diff_xy = _nms_xy(diff, kernel=kernel) 
        nms_diff_xy = nms_diff_xy.squeeze().numpy()
       
        diff_all.append(nms_diff_xy)
    diff_alls = np.stack(diff_all, axis=0) 
    nms_diff_xy = np.max(diff_alls, axis=0)
    nms_diff_xy = torch.as_tensor(nms_diff_xy)
     
    mean_nms = nms_diff_xy[torch.where(nms_diff_xy > 0)].mean().item()
    std_nms_half = nms_diff_xy[torch.where(nms_diff_xy > 0)].std().item() 
    cutoff_score = mean_nms + std_nms_half*0.5
    nms_diff_xy = nms_diff_xy.squeeze().numpy()
    scores, coords = non_maximum_suppression_3d(nms_diff_xy, 14, threshold=cutoff_score)

    return scores, coords
    
def get_potential_coords(rec, sigma1=2, sigma2=4, kernel=3, K = 5000):
    im1 = gaussian_filter(rec, sigma1)
    im2 = gaussian_filter(rec, sigma2)
    diff = im2 - im1 
    diff = torch.as_tensor(diff)
    diff = diff.unsqueeze(0).unsqueeze(0)
    nms_diff = _nms(diff, kernel=kernel)
    topk_scores, topk_zs, topk_ys, topk_xs, topk_inds = _topk(nms_diff, K=K)
    return topk_zs, topk_ys, topk_xs

class FixedRotation(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, img):
        num_rot = np.random.choice(4)
        rot_img = torch.rot90(img, k=num_rot, dims=[1,2])
        return rot_img

class AdjustBrightness(torch.nn.Module):
    def __init__(self, p = 0.5, brightness_factor = 1.2):
        super().__init__()
        self.p = p  
        self.brightness_factor = brightness_factor
    def forward(self, img):
        if torch.rand(1) < self.p: 
            return F.adjust_brightness(img, self.brightness_factor)
        else:
            return img 

class InvertColor(torch.nn.Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p  

    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.invert(img)
        else:
            return img 
# class CenterResizeCrop(torch.nn.Module):
#     def __init__(
#         self,
#         size,
#         scale=(0.08, 1.0),
#         # ratio=(3.0 / 4.0, 4.0 / 3.0),
#         interpolation=InterpolationMode.BILINEAR,
#         antialias: Optional[Union[str, bool]] = "warn",
#     ):
#         super().__init__()
#         if not isinstance(scale, Sequence):
#             raise TypeError("Scale should be a sequence")
#         if (scale[0] > scale[1]):
#             warnings.warn("Scale and ratio should be of kind (min, max)")
#         self.interpolation = interpolation
#         self.antialias = antialias
#         self.scale = scale

#     @staticmethod
#     def get_params(img:Tensor, scale:Tuple[float, float], ratio: Tuple[float, float])-> Tuple[int, int, int, int]:
#         img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
#         mid_h, mid_w = img_h // 2, img_w // 2
#         area = img_h * img_w 

            
class CornerErasing(torch.nn.Module):
    def __init__(self, p=0.5, scale = (0.02, 0.33), ratio = (0.3, 3.3), value=1, inplace = False):
        super().__init__()
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")

        self.p = p 
        self.scale = scale 
        self.ratio = ratio
        self.value = value  
        self.inplace = inplace 

    @staticmethod
    def get_params(img:Tensor, scale:Tuple[float, float], ratio: Tuple[float, float], value:Optional[List[float]]=None)-> Tuple[int, int, int, int, Tensor]:
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        mid_h, mid_w = img_h // 2, img_w // 2
        area = img_h * img_w 
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < mid_h and w < mid_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]
            p1 = np.random.rand()
            p2 = np.random.rand()
            if p1 > 0.5:
                i = torch.randint(0, mid_h - h - 6, size=(1,)).item()
            else:
                i = torch.randint(mid_h+6, img_h - h + 6, size=(1,)).item()
            if p2 > 0.5:
                j = torch.randint(0, mid_w - w - 6, size=(1,)).item()
            else:
                j = torch.randint(mid_w+6, img_w - w + 6, size=(1,)).item()
            return i, j, h, w, v

        return 0, 0, img_h, img_w, img

    def forward(self, img):
        if torch.rand(1) < self.p:
            if isinstance(self.value, (int, float)):
                value = [self.value, ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    "{} (number of input channels)".format(img.shape[-3])
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            return F.erase(img, x, y, h, w, v, self.inplace)
        return img



class RandomCropNoBorder(torch.nn.Module):
    """Crop the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
    but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    @staticmethod
    def get_params(img: Tensor, exclude: int, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h= F._get_image_size(img)
        print('get size', F._get_image_size(img))
        th, tw = output_size

        if h+1 < th or w+1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(exclude, h - th-exclude + 1, size=(1,)).item()
        j = torch.randint(exclude, w - tw-exclude + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, exclude = 0, padding_mode="constant"):
        super().__init__()
        # _log_api_usage_once(self)

        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.exclude = exclude
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.exclude, self.size)

        return F.crop(img, i, j, h, w)


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class CenterOut(torch.nn.Module):
    def __init__(self, fill=0, crop_ratio = [0.4, 0.7]):
        super().__init__()
        self.fill = fill 
        self.crop_ratio = crop_ratio

    def forward(self, img):
        p = np.random.rand()
        if p > 0.8:
            height, width = F._get_image_size(img)
            mid_h, mid_w = height//2, width//2
            crop_dim = int(torch.floor(height * torch.empty(1).uniform_(self.crop_ratio[0], self.crop_ratio[1])).item()//2)*2
            # mid_h, mid_w, mid_crop = height//2, width//2, self.crop_dim//2 
            mid_crop = crop_dim //2
            side_v = mid_h - mid_crop 
            side_h = mid_w - mid_crop
            # left_hw = height - self.crop_dim
            # print('img', img.shape)
            # print('crop_dim', crop_dim)
            center_view = F.crop(img, side_v, side_h, crop_dim, crop_dim)
            padded_view = F.pad(center_view, (side_v, side_h, side_v, side_h))
            # crop_dim_hw = height // 8 
            # left_hw = height - crop_dim_hw * 2 
            # center_view = F.crop(img, crop_dim_hw, crop_dim_hw, left_hw, left_hw)
            # padded_view = F.pad(center_view, (crop_dim_hw, crop_dim_hw, crop_dim_hw, crop_dim_hw))
            return padded_view
        else:
            return img


def change_view(img):
    img_c = np.swapaxes(img,1, 2)
    return img_c
    
def swap_out(img, diameter):
    c, d, h, w = img.shape 
    dd, hh, ww = d//2, h//2, w//2 
    sample_d = list(np.arange(0, dd-diameter[0]*2)) + list(np.arange(dd+diameter[0], d - diameter[0])) 
    sample_h = list(np.arange(0, hh-diameter[1]*2))+ list(np.arange(hh+diameter[1], h - diameter[1])) 
    sample_w = list(np.arange(0, ww-diameter[2]*2))+ list(np.arange(ww+diameter[2], w - diameter[2])) 
    d_s = random.sample(sample_d,2)
    h_s = random.sample(sample_h,2)
    w_s = random.sample(sample_w,2)
    # print(h_s)
    d_s0, d_s1 = d_s[0], d_s[1]
    h_s0, h_s1 = h_s[0], h_s[1]
    w_s0, w_s1 = w_s[0], w_s[1]
    sub_0 = img[0, d_s0:d_s0+diameter[0], h_s0:h_s0+diameter[1], w_s0:w_s0+diameter[2]]
    sub_1 = img[0, d_s1:d_s1+diameter[0], h_s1:h_s1+diameter[1], w_s1:w_s1+diameter[2]]
    img_c = img.copy()
    img_c[0, d_s0:d_s0+diameter[0], h_s0:h_s0+diameter[1], w_s0:w_s0+diameter[2]] = sub_1 
    img_c[0, d_s1:d_s1+diameter[0], h_s1:h_s1+diameter[1], w_s1:w_s1+diameter[2]] = sub_0
    # img[0, d_s:d_s+diameter[0], h_s:h_s+diameter[1], w_s:w_s+diameter[2]] = 0
    return img_c

def drop_out(img, diameter):
    c, d, h, w = img.shape 
    dd, hh, ww = d//2, h//2, w//2 
    # print(diameter[0])
    # print(diameter[1])
    # sample_d = list(np.arange(0, dd-2-diameter[0])) + list(np.arange(dd+2, d - diameter[0])) 
    # sample_h = list(np.arange(0, hh-8-diameter[1]))+ list(np.arange(hh+8, h - diameter[1])) 
    # sample_w = list(np.arange(0, ww-8-diameter[2]))+ list(np.arange(ww+8, w - diameter[2])) 
    sample_d = list(np.arange(0, dd-diameter[0]*2)) + list(np.arange(dd+diameter[0], d - diameter[0])) 
    sample_h = list(np.arange(0, hh-diameter[1]*2))+ list(np.arange(hh+diameter[1], h - diameter[1])) 
    sample_w = list(np.arange(0, ww-diameter[2]*2))+ list(np.arange(ww+diameter[2], w - diameter[2])) 
    d_s = random.sample(sample_d,1)[0]
    h_s = random.sample(sample_h,1)[0]
    w_s = random.sample(sample_w,1)[0]
    img[0, d_s:d_s+diameter[0], h_s:h_s+diameter[1], w_s:w_s+diameter[2]] = 0
    return img

def center_out(img, diameter):
    c, d, h, w = img.shape 
    dd, hh, ww = d//2, h//2, w//2
    center_out = np.zeros(img.shape)
    center_out[0, :,hh-diameter[1]:hh+diameter[1],ww-diameter[2]:ww+diameter[2]] = img[0,:,hh-diameter[1]:hh+diameter[1],ww-diameter[2]:ww+diameter[2]]
    return center_out

def flip_ud(img, expand=False):
    if expand:
        return np.flip(img, 2).copy()
    else:
        return np.flip(img, 1).copy()

def flip_lr(img, expand=False):
    if expand:
        return np.flip(img, 3).copy()
    else:
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
