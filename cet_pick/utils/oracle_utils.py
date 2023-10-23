from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba

@numba.jit(nopython=True, nogil=True)
def gen_oracle_map(feat, ind, w, h):

	pass