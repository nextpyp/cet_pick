from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from cet_pick.utils.utils import AverageMeter
from cet_pick.models.decode import tomo_decode
from cet_pick.models.utils import _center_distance
from cet_pick.models.loss import SupConLoss
from cet_pick.models.data_parallel import DataParallel
from cet_pick.utils.utils import adjust_learning_rate, warmup_learning_rate, compute_ramped_lrate, adjust_lr_denoise


class ModelWithLossSelfLabel(torch.nn.Module):
    def __init__(self, model, loss, if_ema = False):

        pass

class ModelWithLossDenoise(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLossDenoise, self).__init__()
        self.model = model 
        self.loss = loss 

    def forward(self, batch, epoch, phase):
        noisy_in = batch["noisy_in"]
        if phase != 'train':
            with torch.no_grad():
                net_out = self.model["denoise"](batch["noisy_in"])
                param_est_out = self.model["sigma"](batch["noisy_in"])
        else:
            net_out = self.model["denoise"](batch["noisy_in"])
            param_est_out = self.model["sigma"](batch["noisy_in"])

        
        param_est_out = torch.mean(param_est_out, dim=(2,3), keepdim=True)
        noise_est_out = param_est_out 
        softplus = torch.nn.Softplus()  # yes this line is necessary, don't ask
        noise_est_out = softplus(noise_est_out - 4.0) + 1e-3
        noise_std = noise_est_out 
        # loss, loss_stats = self.loss(net_out, noise_std)
        outputs = {}
        mu_x = net_out[:, 0:1, ...]
        A_c = net_out[:, 1:2, ...]
        sigma_x = A_c ** 2 
        sigma_n = noise_std ** 2  
        sigma_y = sigma_x + sigma_n 
        pme_out = (noisy_in * sigma_x + mu_x * sigma_n) / (sigma_x + sigma_n)
        loss, loss_stats = self.loss(noisy_in, mu_x, sigma_y, noise_std)
        net_std_out = (sigma_x ** 0.5)[:,0,...]
        noise_std_out = noise_std[:,0,...]
        outputs["img_mu"] = mu_x 
        outputs["img_denoise"] = pme_out 
        outputs["model_std"] = net_std_out
        return outputs, loss, loss_stats

class ModelWithLossSCAN2D3D(torch.nn.Module):
    def __init__(self, model, loss, update_cluster_head_only = False):
        super(ModelWithLossSCAN2D3D, self).__init__()
        if update_cluster_head_only:
            model = model.eval()
        else:
            model = model.train()
        self.model = model 
        self.loss = loss 
        self.update_cluster_head_only = update_cluster_head_only

    def forward(self, batch, epoch, phase):
        if phase == 'train':
            if self.update_cluster_head_only:
                with torch.no_grad():
                    anchors_features = self.model(batch['anchor_2d'], batch['anchor_3d'], forward_pass='backbone')
                    neighbor_features = self.model(batch['neighbor_2d'], batch['neighbor_3d'], forward_pass='backbone')
                anchors_output = self.model(anchors_features, None, forward_pass='head')
                neighbors_output = self.model(neighbor_features, None, forward_pass='head')
            else: 
                anchors_output = self.model(batch['anchor_2d'], batch['anchor_3d'])
                neighbors_output = self.model(batch['neighbor_2d'], batch['neighbor_3d'])
            loss, loss_stats = self.loss(anchors_output, neighbors_output, batch, epoch)
            return [anchors_output, neighbors_output], loss, loss_stats
        
class ModelWithLossSCAN(torch.nn.Module):
    def __init__(self, model, loss, update_cluster_head_only = False):
        super(ModelWithLossSCAN, self).__init__()
        if update_cluster_head_only:
            model = model.eval()
        else:
            model = model.train()
        self.model = model 
        self.loss = loss 
        self.update_cluster_head_only = update_cluster_head_only

    def forward(self, batch, epoch, phase):
        if phase == 'train':
            if self.update_cluster_head_only:
                with torch.no_grad():
                    anchors_features = self.model(batch['anchor'], forward_pass='backbone')
                    neighbor_features = self.model(batch['neighbor'], forward_pass='backbone')
                anchors_output = self.model(anchors_features, forward_pass='head')
                neighbors_output = self.model(neighbor_features, forward_pass='head')
            else: 
                anchors_output = self.model(batch['anchor'])
                neighbors_output = self.model(batch['neighbor'])
            loss, loss_stats = self.loss(anchors_output, neighbors_output, batch, epoch)
            return [anchors_output, neighbors_output], loss, loss_stats
        # elif phase == 'val':


class ModelWithLossSimSiam2D3D(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLossSimSiam2D3D, self).__init__()
        self.model = model 
        self.loss = loss  
    def forward(self, batch, epoch, phase):
        outputs = self.model(batch['input'], batch['input_3d'], batch['input_aug'], batch['input_aug_3d'])
        loss, loss_stats = self.loss(outputs, batch, epoch)

        return outputs, loss, loss_stats



class ModelWithLossSimSiam(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLossSimSiam, self).__init__()
        self.model = model 
        self.loss = loss  
    def forward(self, batch, epoch, phase):
        outputs = self.model(batch['input'], batch['input_aug'])
        loss, loss_stats = self.loss(outputs, batch, epoch)

        return outputs, loss, loss_stats

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model 
        self.loss = loss 

    def forward(self, batch, epoch, phase):
        # this part is only for contrastive use comment out if we are not doing contrastive learning
        if phase == 'train':
            print('batch input', batch['input'].shape)
            outputs = self.model(batch['input'])
            outputs_cr = self.model(batch['input_aug'])
            loss, loss_stats = self.loss(outputs, batch, epoch, phase, output_cr=outputs_cr)
        else:
            with torch.no_grad():
                self.model.eval()
                # if epoch == 10:
                #     print('-------------val phase model weights------------')
                #     for name, param in self.model.named_parameters():
                #         print(name, param.data)
                # # print('val model', self.model)
                print('woc?')
                print('batch input shape', batch['input'].shape)
                outputs = self.model(batch['input'])
                outputs_cr = None
                loss, loss_stats = self.loss(outputs, batch, epoch, phase, output_cr=outputs_cr)
        return outputs[-1], loss, loss_stats 


class ModelWithLossClass(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLossClass, self).__init__()
        self.model = model 
        self.loss = loss 

    def fill(self):
        self.model.fill()

    def unfill(self):
        self.model.unfill()

    def forward(self, batch, epoch, phase):
        # this part is only for contrastive use comment out if we are not doing contrastive learning
        if phase == 'train':

            outputs = self.model(batch['input'])
            outputs_cr = self.model(batch['input_aug'])
            loss, loss_stats = self.loss(outputs, batch, epoch, phase, output_cr=outputs_cr)
        else:
            with torch.no_grad():

                outputs = self.model(batch['input'])
                outputs_cr = None
                loss, loss_stats = self.loss(outputs, batch, epoch, phase, output_cr=outputs_cr)
        return outputs[-1], loss, loss_stats 

class ModelWithLossCl(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLossCl, self).__init__()
        self.model = model 
        self.loss = loss 

    def forward(self, batch, epoch, phase, cluster_centers = None, cluster_ind = None):

        outputs = self.model(batch['input'])
        # this part is only for contrastive use comment out if we are not doing contrastive learning
        if phase == 'train':
            outputs_cr = self.model(batch['input_aug'])
        # else:
        #     outputs_cr = None
        loss, loss_stats, cluster_centers, cluster_ind = self.loss(outputs, batch, epoch, phase, outputs_cr=outputs_cr, cluster_center = cluster_centers, cluster_ind = cluster_ind)
        return outputs[-1], loss, loss_stats, cluster_centers, cluster_ind




class BaseTrainer(object):
    def __init__(
        self, opt, model, optimizer=None):
        self.opt= opt 
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.update_cluster_head_only = opt.cluster_head 
        self.iter = 0
        if opt.task == 'simsiam' or opt.task == 'moco' or opt.task == 'simsiam3d':
            self.model_with_loss = ModelWithLossSimSiam(model, self.loss)
        elif opt.task == 'simsiam2d3d':
            self.model_with_loss = ModelWithLossSimSiam2D3D(model, self.loss)
        elif opt.task == 'scan':
            self.model_with_loss = ModelWithLossSCAN(model,self.loss,update_cluster_head_only=self.update_cluster_head_only)
        elif opt.task == 'scan2d3d':
            self.model_with_loss = ModelWithLossSCAN2D3D(model,self.loss,update_cluster_head_only=self.update_cluster_head_only)
        elif opt.task == 'denoise':
            self.model_with_loss = ModelWithLossDenoise(model, self.loss)
        elif opt.task == 'semiclass':
            self.model_with_loss = ModelWithLossClass(model, self.loss)
        else:
            self.model_with_loss = ModelWithLoss(model, self.loss)
        # self.model_with_loss = ModelWithLossCl(model, self.loss)

    def set_distributed_device(self, gpus):
        if gpus is not None:
            torch.cuda.set_device(gpus)
            self.model_with_loss.cuda(gpus)
            self.model_with_loss = torch.nn.parallel.DistributedDataParallel(self.model_with_loss, device_ids = [gpus])
            model_without_ddp = self.model_with_loss.module 
        else:
            self.model_with_loss.cuda()
            self.model_with_loss = torch.nn.parallel.DistributedDataParallel(self.model_with_loss)
            model_without_ddp = self.model_with_loss.module

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids = gpus, chunk_sizes = chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch_contrastive(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt 
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters 
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):

            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            
            for k in batch:
                if k!= 'meta':
                    batch[k] = batch[k].to(device = opt.device, non_blocking=True)
        return

    def run_epoch_fewshot(self, phase, epoch, data_loader, cluster_centers = None, cluster_ind = None):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()
 
        opt = self.opt 
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters 

        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            
            for k in batch:

                if k!= 'meta' and k!='ml_graph' and k!='cl_graph':
                    batch[k] = batch[k].to(device = opt.device, non_blocking=True)
            output, loss, loss_stats, cluster_centers, cluster_ind = model_with_loss(batch, epoch, phase, cluster_centers, cluster_ind)
            loss = loss.mean()
            # torch.autograd.set_detect_anomaly(True)
            if phase == 'train':
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
            else:
                bar.next()
            if phase != 'train':
                if opt.debug > 0:
                    self.debug(batch, output, cluster_centers[cluster_ind], iter_id)

            if opt.test:
                self.save_result(output, batch, cluster_centers[cluster_ind], results)

            del output, loss, loss_stats
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.


        return ret, results, cluster_centers, cluster_ind



    def run_epoch_denoise(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if self.opt.task != 'scan':
            if phase == 'train':
                model_with_loss.train()
            else:
                if len(self.opt.gpus) > 1:
                    model_with_loss = self.model_with_loss.module
                model_with_loss.eval()
                torch.cuda.empty_cache()
        if self.opt.task == 'scan':
            if phase == 'train':
                if not self.update_cluster_head_only:
                    model_with_loss.train()
            else:
                if len(self.opt.gpus) > 1:
                    model_with_loss = self.model_with_loss.module
                model_with_loss.eval()
                torch.cuda.empty_cache()


        opt = self.opt 
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters 
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            # actual_iter = iter
            adjust_lr_denoise(self.iter+1, opt.num_iters, 0.2, 0.7, opt.lr, self.optimizer)
            self.iter = self.iter + opt.batch_size

            if self.iter >= num_iters:
                break
            data_time.update(time.time() - end)
            
            for k in batch:
                if k!= 'meta':
                    if opt.distributed:
                    # batch[k] = batch[k].to(device = opt.device, non_blocking=True)
                        batch[k] = batch[k].cuda(opt.gpu)
                    else:
                        batch[k] = batch[k].to(device = opt.device, non_blocking=True)
  
            output, loss, loss_stats = model_with_loss(batch, epoch, phase)
            
            if self.opt.task == 'denoise':
                loss = loss.mean()

            if phase == 'train':
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, self.iter, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                if opt.task == 'scan':

                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['anchor'].size(0))
                elif opt.task == 'scan2d3d':
                    
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['anchor_2d'].size(0))
                elif opt.task == 'denoise':
                    
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['noisy_in'].size(0))
                else:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
            else:
                bar.next()
            if phase != 'train':
                if opt.debug > 0:

                    self.debug(batch, output, iter_id)

            if opt.test:
                self.save_result(output, batch, results)

            del output, loss, loss_stats
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.


        return ret, results 

    def run_epoch(self, phase, epoch, data_loader):

        model_with_loss = self.model_with_loss
        if self.opt.task != 'scan':
            if phase == 'train':
                model_with_loss.train()
            else:
                if len(self.opt.gpus) > 1:
                    model_with_loss = self.model_with_loss.module
                model_with_loss.eval()
                torch.cuda.empty_cache()
        if self.opt.task == 'scan':
            if phase == 'train':
                if not self.update_cluster_head_only:
                    model_with_loss.train()
            else:
                if len(self.opt.gpus) > 1:
                    model_with_loss = self.model_with_loss.module
                model_with_loss.eval()
                torch.cuda.empty_cache()
        if self.opt.task == 'semiclass':
            if phase == 'train':
                model_with_loss.train()
                model_with_loss.unfill()
            else:
                if len(self.opt.gpus) > 1:
                    model_with_loss = self.model_with_loss.module
                model_with_loss.eval()
                model_with_loss.fill()
                torch.cuda.empty_cache()


        opt = self.opt 
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters 
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):

            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            
            for k in batch:
                if k!= 'meta':
                    if opt.distributed:
                    # batch[k] = batch[k].to(device = opt.device, non_blocking=True)
                        batch[k] = batch[k].cuda(opt.gpu)
                    else:
                        batch[k] = batch[k].to(device = opt.device, non_blocking=True)
                    # batch[k] = batch[k].to(device = opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch, epoch, phase)
            
            if self.opt.task == 'denoise':
                loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                if opt.task == 'scan':

                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['anchor'].size(0))
                elif opt.task == 'scan2d3d':
                    
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['anchor_2d'].size(0))
                elif opt.task == 'denoise':
                    
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['noisy_in'].size(0))
                else:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
            else:
                bar.next()
            if phase != 'train':
                if opt.debug > 0:
                    self.debug(batch, output, epoch)

            if opt.test:
                self.save_result(output, batch, results)

            del output, loss, loss_stats
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.


        return ret, results 

    def debug(self, batch, output, iter_id):
        raise NotImplementedError 

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError 

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)
        # return self.run_epoch_fewshot('val', epoch, data_loader, cluster_centers, cluster_ind)


    def train(self, epoch, data_loader):
        # if self.opt.task == 'denoise':
        #     return self.run_epoch_denoise('train', epoch, data_loader)
        # else:
        return self.run_epoch('train', epoch, data_loader)
        # return self.run_epoch_fewshot('train', epoch, data_loader, cluster_centers, cluster_ind)


