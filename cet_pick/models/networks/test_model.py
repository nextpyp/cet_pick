from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import torch
import torch.nn as nn
from cet_pick.models.utils import insize_from_outsize_3d, out_from_in
from cet_pick.models.networks.resnet_new import ResNet8_last3D, ResNet8_more3D

model = ResNet8_more3D(heads={'hm': 16}, head_conv=None, bn=False)
print('model', model)
print('width', model.width_z, model.width_xy)
print('fill model...')
model.fill()
print('model', model)
print('unfill model...')
model.unfill()
print('model', model)