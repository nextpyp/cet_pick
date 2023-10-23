from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from cet_pick.models.loss import FocalLoss, RegL1Loss, RegLoss, BCELoss
from cet_pick.models.decode import tomo_decode
from cet_pick.models.decode import _nms
from cet_pick.models.utils import _sigmoid 
from cet_pick.trains.base_trainer import BaseTrainer 
from cet_pick.utils.debugger import Debugger
from cet_pick.utils.post_process import tomo_post_process
import cv2

class TomoLoss(torch.nn.Module):
	def __init__(self, opt):
		super(TomoLoss, self).__init__()
		self.bce = BCELoss()


		self.opt = opt 

	def forward(self, outputs, batch, epoch, phase, output_cr = None):
		opt = self.opt 
		# hm_loss, wh_loss, off_loss = 0, 0, 0 
		bce_loss = 0
		for s in range(opt.num_stacks):
			output = outputs[s]
			# if not opt.mse_loss:
				# output['hm'] = _sigmoid(output['hm'])
				# output['class'] = 
		bce_loss += self.bce(output['class'], batch['class'])

		loss_stats = {'loss': bce_loss}

		return bce_loss, loss_stats

class TomoClassTrainer(BaseTrainer):
	def __init__(self, opt, model, optimizer=None):
		super(TomoClassTrainer, self).__init__(opt, model, optimizer=optimizer)

	def _get_losses(self, opt):
		# loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
		loss_states = ['loss']
		loss = TomoLoss(opt)
		return loss_states, loss 

	def debug(self, batch, output, iter_id):
		opt = self.opt 
		class_result = output['class']
		gt_result = batch['class']

		
		for i in range(1):
			debugger = Debugger(dataset = opt.dataset, down_ratio = opt.down_ratio)
			tomo = batch['input'][i].detach().cpu().numpy()
			
			# preds = output['class'][i].flatten().detach().cpu().numpy()
			# gts = batch[]
					# 	gts = post_dets_gt[i]
		# 	preds = post_dets[i]
		# 	det_zs = preds.keys()
		# 	# print(det_zs)
			# out_z = output['class'].flatten().detach().cpu().numpy()
			out_z = output['class'].flatten().detach().cpu().numpy()
			# print('out_z', out_z)

			out_z_gt = batch['class'][i].flatten().detach().cpu().numpy()
			for z in range(16):
				
				
				out = out_z[z]
				# print('out',out)
				if out > 0:
		
					tomo_z = cv2.normalize(tomo[z], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
					tomo_z = np.clip(tomo_z * 255., 0, 255).astype(np.uint8)
					img_slice = np.dstack((tomo_z, tomo_z, tomo_z))

					debugger.add_slice(img_slice, img_id = str(iter_id))

					if opt.debug == 4:
						debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id), slice_num = z)


	def save_results(self, output, batch, results):
		reg = output['reg'] if self.opt.reg_offset else None 
		dets = tomo_decode(output['hm'], output['wh'], reg=reg, K = self.opt.K)
		dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
		return dets 