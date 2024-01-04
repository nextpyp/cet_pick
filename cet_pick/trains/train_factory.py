from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cet_pick.trains.tomo_trainer import TomoTrainer 
from cet_pick.trains.tomo_classifier_trainer import TomoClassTrainer
from cet_pick.trains.tomo_cr_trainer import TomoCRTrainer
from cet_pick.trains.tomo_fewshot_cr_trainer import TomoKMTrainer
from cet_pick.trains.tomo_cr_semi_trainer import TomoCRSemiTrainer
from cet_pick.trains.tomo_simsiam_trainer import TomoSimSiamTrainer
from cet_pick.trains.tomo_moco_trainer import TomoMocoTrainer
from cet_pick.trains.tomo_scan_trainer import TomoSCANTrainer
from cet_pick.trains.tomo_denoise_trainer import TomoDenoiseTrainer

train_factory = {
	'tomo':TomoTrainer,
	'tcla': TomoClassTrainer,
	'cr':TomoCRTrainer,
	'fs':TomoKMTrainer,
	'semi':TomoCRSemiTrainer,
	'semi3d':TomoCRSemiTrainer,
	'simsiam':TomoSimSiamTrainer,
	'moco': TomoMocoTrainer,
	'scan': TomoSCANTrainer,
	'simsiam2d3d': TomoSimSiamTrainer,
	'simsiam3d':TomoSimSiamTrainer,
	'scan2d3d': TomoSCANTrainer,
	'denoise': TomoDenoiseTrainer
}