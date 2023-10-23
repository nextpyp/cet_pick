from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cet_pick.datasets.tomos import TOMO 
from cet_pick.datasets.particle import ParticleDataset 
from cet_pick.datasets.tomo_moco import TOMOMoco
from cet_pick.datasets.particle_moco import ParticleMocoDataset
from cet_pick.datasets.tomo_moco_3d import TOMOMoco3D
from cet_pick.datasets.particle_moco_3d import ParticleMocoDataset3D
from cet_pick.datasets.tomo_fewshot import TOMOFewShot 
from cet_pick.datasets.particle_fewshot import ParticleFewShotDataset

dataset_factory = {
	'tomo': TOMO,
	'cr': TOMOMoco,
	'fs':TOMOFewShot,
	'semi': TOMOMoco,
	'semi_test': TOMO,
	'semi3d': TOMOMoco3D
}

_sample_factory = {
	'tomo': ParticleDataset,
	'tcla': ParticleDataset,
	'cr': ParticleMocoDataset,
	'fs': ParticleFewShotDataset,
	'semi':ParticleMocoDataset,
	'semi_test': ParticleDataset,
	'semi3d':ParticleMocoDataset3D

}


def get_dataset(dataset, task):
	class Dataset(dataset_factory[dataset], _sample_factory[task]):
		pass 
	return Dataset