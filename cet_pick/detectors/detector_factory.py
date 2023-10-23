from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cet_pick.detectors.tomo_det import TomodetDetector 

detector_factory = {
	'tomo': TomodetDetector,
	'semi': TomodetDetector,
	'semi3d': TomodetDetector
}