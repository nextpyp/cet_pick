from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random
import mrcfile
from cet_pick.utils.loader import load_rec
from scipy.ndimage import laplace
from scipy import fftpack 
rec = load_rec('/nfs/bartesaghilab2/qh36/joint_data/lip_channel/test_files/795_1.rec')
print('rec', rec.shape)
# for sl in range(rec.shape[0]):
	# single_sl = rec[sl]
	# fft_sl = fftpack.fft2(single_sl)
	# lp = laplace(single_sl)
	# print(sl)
	# print(fft_sl[1, 0], fft_sl[0,1], fft_sl[2,0], fft_sl[0,2])

	# print(np.sum(lp))
	# print(lp)