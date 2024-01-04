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
from cet_pick.datasets.tomo_classify_moco import TOMOMocoClass 
from cet_pick.datasets.particle_moco_classify import ParticleMocoClassDataset
from cet_pick.datasets.particle_pre import ParticlePreDataset
from cet_pick.datasets.tomo_pre import TOMOPre
from cet_pick.datasets.tomo_pre_test import TOMOPreTest
from cet_pick.datasets.tomo_pre_shrec import TOMOPreSHREC
from cet_pick.datasets.tomo_pre_2d import TOMOPre2D 
from cet_pick.datasets.particle_pre_2d import ParticlePre2DDataset
from cet_pick.datasets.tomo_pre_proj import TOMOPreProj
from cet_pick.datasets.tomo_pre_proj_angle import TOMOPreProjAngle
from cet_pick.datasets.tomo_pre_proj_angle_select_new import TOMOPreProjAngleSelect
from cet_pick.datasets.tomo_pre_test_angle import TOMOPreTestAngle
from cet_pick.datasets.particle_pre_2d_proj_new import ParticlePreProjDataset
from cet_pick.datasets.tomo_scan_proj_angle_select import TOMOSCANProjAngleSelect
from cet_pick.datasets.particle_scan_2d_proj import ParticleSCANProjDataset
from cet_pick.datasets.tomo_pre_proj_angle_select_new2d3d import TOMOPreProjAngleSelect2D3D
from cet_pick.datasets.particle_pre_2d_proj_new2d3d import ParticlePreProjDataset2D3D
from cet_pick.datasets.particle_pre_2d_proj_new3donly import ParticlePreProjDataset3D
from cet_pick.datasets.tomo_pre_proj_angle_select_3donly import TOMOPreProjAngleSelect3D
from cet_pick.datasets.tomo_scan_proj_angle_select_2d3d import TOMOSCAN2D3DProjAngleSelect
from cet_pick.datasets.particle_scan_2d3d_proj import ParticleSCAN2D3DProjDataset
from cet_pick.datasets.tomo_pre_proj_angle_select_new3d_vol import TOMOPreProjAngleSelect3DVol
from cet_pick.datasets.particle_pre_3d_vol import ParticlePre3DVolDataset
from cet_pick.datasets.tomo_denoise import TOMODenoise 
from cet_pick.datasets.particle_denoise import ParticleDenoiseDataset
from cet_pick.datasets.tomo_post_proj_angle_select import TOMOPostProjAngleSelect3DVol

dataset_factory = {
	'tomo': TOMO,
	'cr': TOMOMoco,
	'fs':TOMOFewShot,
	'semi': TOMOMoco,
	'semi_test': TOMO,
	'semi3d': TOMOMoco3D,
	'semiclass': TOMOMocoClass,
	'simsiam': TOMOPre,
	'simsiam_test': TOMOPreTest,
	'simsiam2d': TOMOPre2D,
	'simsiamproj': TOMOPreProjAngleSelect,
	'simsiamproj_test':TOMOPreTestAngle,
	'scan': TOMOSCANProjAngleSelect,
	'simsiam2d3d': TOMOPreProjAngleSelect2D3D,
	# 'simsiam3d': TOMOPreProjAngleSelect3D,
	'simsiam3d':TOMOPreProjAngleSelect3DVol,
	'simsiam3dpost': TOMOPostProjAngleSelect3DVol,
	'scan2d3d': TOMOSCAN2D3DProjAngleSelect,
	'denoise': TOMODenoise
}

_sample_factory = {
	'tomo': ParticleDataset,
	'tcla': ParticleDataset,
	'cr': ParticleMocoDataset,
	'fs': ParticleFewShotDataset,
	'semi':ParticleMocoDataset,
	'semi_test': ParticleDataset,
	'semi3d':ParticleMocoDataset3D,
	'semiclass':ParticleMocoClassDataset,
	'simsiam':ParticlePreDataset,
	'simsiam2d':ParticlePre2DDataset,
	'simsiamproj': ParticlePreProjDataset,
	'scan':ParticleSCANProjDataset,
	'simsiam2d3d': ParticlePreProjDataset2D3D,
	'simsiam3dpost': ParticlePreProjDataset3D,
	'simsiam3d':ParticlePre3DVolDataset,
	'scan2d3d': ParticleSCAN2D3DProjDataset,
	'denoise': ParticleDenoiseDataset

}


def get_dataset(dataset, task):
	class Dataset(dataset_factory[dataset], _sample_factory[dataset]):
		pass 
	return Dataset