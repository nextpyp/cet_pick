from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cet_pick.trains.tomo_trainer import TomoTrainer 
from cet_pick.trains.tomo_classifier_trainer import TomoClassTrainer
from cet_pick.trains.tomo_cr_trainer import TomoCRTrainer
from cet_pick.trains.tomo_fewshot_cr_trainer import TomoKMTrainer
from cet_pick.trains.tomo_cr_semi_trainer import TomoCRSemiTrainer

train_factory = {
	'tomo':TomoTrainer,
	'tcla': TomoClassTrainer,
	'cr':TomoCRTrainer,
	'fs':TomoKMTrainer,
	'semi':TomoCRSemiTrainer,
	'semi3d':TomoCRSemiTrainer
}