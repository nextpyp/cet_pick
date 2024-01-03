from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
# from utils.loader import load_subtomo,load_subtomo_v2
from utils.loader import load_rec
from cet_pick.datasets.dataset_factory import dataset_factory
from cet_pick.detectors.detector_factory import detector_factory
from utils.image import gaussian_radius, draw_umich_gaussian_3d, draw_msra_gaussian_3d, flip_ud, flip_lr, CornerErasing, CenterOut
import torchio as tio 
from cet_pick.utils.loader import load_tomos_from_list, cutup
from cet_pick.models.model import create_model, load_model 
from cet_pick.utils.memory_bank import MemoryBank
from cet_pick.trains.tomo_moco_small_trainer import MoCoModel, MoCoTrainer
from sklearn.cluster import KMeans
import mrcfile
import torch.nn.functional as F
import torchvision.transforms as T
from cet_pick.utils.lie_tools import random_SO3, constrained_SO3
from cet_pick.utils.project3d import Projector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PrefetchDatasetProj(torch.utils.data.Dataset):

    def __init__(self, opt, dataset):
        self.opt = opt 
        # self.images = dataset.images 
        # self.images = dataset.tomos
        # self.image_paths = dataset.subtomo_paths 
        self.tomos = dataset.tomos 
        self.subvols = dataset.sub_vols_3d
        # self.labels = dataset.labels
        self.mean_subvols = dataset.mean_subvols3d
        self.std_subvols = dataset.std_subvols3d
        self.names = dataset.names_all
        self.coords = dataset.coords
        self.size = dataset.size
        self.transforms = T.Compose([
            T.ToPILImage(),
            # T.CenterCrop(self.size[1]),
            T.ToTensor(),
            # CenterOut(crop_dim = 18),
            T.Normalize((self.mean_subvols),(self.std_subvols))]
            # T.Normalize((0.5),(0.5))]
            )
        # self.hm_shape = dataset.hm_shape
        # rot = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
        # self.rot = rot.unsqueeze(0).float()
        # self.size = dataset.size
        # single_tomo = self.tomos[0]
        # blks = cutup(single_tomo[15:20],dataset.size, (2,2,2))
        # self.blks = []
        # for i in range(blks.shape[0]):
        #     for j in range(blks.shape[1]):
        #         for k in range(blks.shape[2]):
        #             self.blks.append(blks[i,j,k])
        # del blks

        # self.labels = dataset.labels
        # self.targets = dataset.targets 
        # self.targets = dataset.gt_dets
        # self.names = dataset.names
    # def _downscale_coord(self,ann):
    #     x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]
    #     return [x,y,z]

    def __getitem__(self, index): 

        # crop = tio.transforms.Crop((self.size[0]//4, self.size[1]//4, self.size[2]//4))
        # img = self.images[index]
        # subtomo_path = self.image_paths[index]
        # label = self.labels[index]
        coord = self.coords[index]
        name = self.names[index]
        # print('coord',coord)
        # subtomo = load_subtomo_v2(subtomo_path)
        subtomo = self.subvols[index]
        # print('subtomo', subtomo.shape)
        # subtomo = np.mean(subtomo, axis=0)
        # subtomo -= subtomo.min()
        # subtomo /= subtomo.max()
        # subtomo = torch.tensor(subtomo)
        # subtomo = subtomo.unsqueeze(0).float()
        # zz = self.size[0]//2 
        # sub_vol_c = np.zeros(subtomo.shape)
        # sub_vol_c[zz-4:zz+4] = subtomo[zz-4:zz+4]
        # projector = Projector(sub_vol_c)
        # proj = projector.project(self.rot)
        normed_tomo = self.transforms(subtomo)
        # if subtomo.shape[1] != 48 or subtomo.shape[2] != 48:
        #     print('subtomo', subtomo.shape)
        #     print('label', label)
        # subtomo = np.expand_dims(subtomo,axis=0).copy()
        # coords = self.targets[index]
        # name = self.names[index]
        # gt_det = []
        # num_objs = len(coords)
        # for k in range(num_objs):
        #     ann = coords[k]
        #     ann = self._downscale_coord(ann)
        #     # print('ann', ann)
        #     ann = np.array(ann)
        #     gt_det.append(ann)
        # normed_tomo = norm(subtomo)
        # cropped_tomo = crop(normed_tomo)
        # print('cropped_tomo', cropped_tomo.shape)
        # normed_tomo = normed_tomo.squeeze().transpose(-1,0,1)
        # gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
        ret = {'input':normed_tomo.float(), 'coord': coord, 'label': 0, 'name': name, 'orig': subtomo}
        # meta = {'name': name}
        # ret['meta'] = meta
        return ret 

    def __len__(self):
        return len(self.subvols)


class PrefetchDataset2D(torch.utils.data.Dataset):

    def __init__(self, opt, dataset):
        self.opt = opt 
        # self.images = dataset.images 
        # self.images = dataset.tomos
        # self.image_paths = dataset.subtomo_paths 
        self.tomos = dataset.tomos 
        self.subvols = dataset.subvols
        self.labels = dataset.labels
        self.size = dataset.size
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(self.size[1]-self.size[1]//4),
            T.ToTensor(),
            T.Normalize((0.5),(0.5))]
            )
        rot = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
        self.rot = rot.unsqueeze(0).float()
        # self.size = dataset.size
        # single_tomo = self.tomos[0]
        # blks = cutup(single_tomo[15:20],dataset.size, (2,2,2))
        # self.blks = []
        # for i in range(blks.shape[0]):
        #     for j in range(blks.shape[1]):
        #         for k in range(blks.shape[2]):
        #             self.blks.append(blks[i,j,k])
        # del blks

        # self.labels = dataset.labels
        # self.targets = dataset.targets 
        # self.targets = dataset.gt_dets
        # self.names = dataset.names
    # def _downscale_coord(self,ann):
    #     x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]
    #     return [x,y,z]

    def __getitem__(self, index): 

        # crop = tio.transforms.Crop((self.size[0]//4, self.size[1]//4, self.size[2]//4))
        # img = self.images[index]
        # subtomo_path = self.image_paths[index]
        label = self.labels[index]
        # subtomo = load_subtomo_v2(subtomo_path)
        subtomo = self.subvols[index]
        zz = self.size[0]//2 
        sub_vol_c = np.zeros(subtomo.shape)
        sub_vol_c[zz-4:zz+4] = subtomo[zz-4:zz+4]
        projector = Projector(sub_vol_c)
        proj = projector.project(self.rot)
        normed_tomo = self.transforms(proj)
        # if subtomo.shape[1] != 48 or subtomo.shape[2] != 48:
        #     print('subtomo', subtomo.shape)
        #     print('label', label)
        # subtomo = np.expand_dims(subtomo,axis=0).copy()
        # coords = self.targets[index]
        # name = self.names[index]
        # gt_det = []
        # num_objs = len(coords)
        # for k in range(num_objs):
        #     ann = coords[k]
        #     ann = self._downscale_coord(ann)
        #     # print('ann', ann)
        #     ann = np.array(ann)
        #     gt_det.append(ann)
        # normed_tomo = norm(subtomo)
        # cropped_tomo = crop(normed_tomo)
        # print('cropped_tomo', cropped_tomo.shape)
        # normed_tomo = normed_tomo.squeeze().transpose(-1,0,1)
        # gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
        ret = {'input':normed_tomo.float(), 'label': label}
        # meta = {'name': name}
        # ret['meta'] = meta
        return ret 

    def __len__(self):
        return len(self.subvols)


class PrefetchDataset(torch.utils.data.Dataset):

    def __init__(self, opt, dataset):
        self.opt = opt 
        # self.images = dataset.images 
        # self.images = dataset.tomos
        # self.image_paths = dataset.subtomo_paths 
        self.tomos = dataset.tomos 
        self.subvols = dataset.subvols
        self.labels = dataset.labels
        self.size = dataset.size
        # self.size = dataset.size
        # single_tomo = self.tomos[0]
        # blks = cutup(single_tomo[15:20],dataset.size, (2,2,2))
        # self.blks = []
        # for i in range(blks.shape[0]):
        #     for j in range(blks.shape[1]):
        #         for k in range(blks.shape[2]):
        #             self.blks.append(blks[i,j,k])
        # del blks

        # self.labels = dataset.labels
        # self.targets = dataset.targets 
        # self.targets = dataset.gt_dets
        # self.names = dataset.names
    # def _downscale_coord(self,ann):
    #     x, y, z = ann[0]//self.opt.down_ratio, ann[1]//self.opt.down_ratio, ann[2]
    #     return [x,y,z]

    def __getitem__(self, index): 
        norm = tio.Compose([
            tio.transforms.Crop((self.size[0]//8, self.size[1]//8, self.size[2]//8)),
            tio.transforms.ZNormalization(),
            tio.transforms.RescaleIntensity(out_min_max=(-3,3)),
            tio.transforms.ZNormalization()
                ]
            )
        # crop = tio.transforms.Crop((self.size[0]//4, self.size[1]//4, self.size[2]//4))
        # img = self.images[index]
        # subtomo_path = self.image_paths[index]
        label = self.labels[index]
        # subtomo = load_subtomo_v2(subtomo_path)
        subtomo = self.subvols[index]
        # if subtomo.shape[1] != 48 or subtomo.shape[2] != 48:
        #     print('subtomo', subtomo.shape)
        #     print('label', label)
        subtomo = np.expand_dims(subtomo,axis=0).copy()
        # coords = self.targets[index]
        # name = self.names[index]
        # gt_det = []
        # num_objs = len(coords)
        # for k in range(num_objs):
        #     ann = coords[k]
        #     ann = self._downscale_coord(ann)
        #     # print('ann', ann)
        #     ann = np.array(ann)
        #     gt_det.append(ann)
        normed_tomo = norm(subtomo)
        # cropped_tomo = crop(normed_tomo)
        # print('cropped_tomo', cropped_tomo.shape)
        # normed_tomo = normed_tomo.squeeze().transpose(-1,0,1)
        # gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1,3), dtype=np.float32)
        ret = {'input':normed_tomo, 'label': label}
        # meta = {'name': name}
        # ret['meta'] = meta
        return ret 

    def __len__(self):
        return len(self.subvols)

        


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    # Detector = detector_factory[opt.task]
    # model = create_model(opt.arch, opt.heads, opt.head_conv)
    model_q = create_model(opt.arch, opt.heads, opt.head_conv)
    model_k = create_model(opt.arch, opt.heads, opt.head_conv)
    model = MoCoModel(model_q, model_k,opt)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()
    model = model.encoder_q
    print('model', model)
    # detector = Detector(opt)
    # split = 'val' if not opt.trainval else 'test'
    split = 'test'

    dataset = Dataset(opt, split,(3,opt.bbox,opt.bbox), sigma1=[2.5,5], K = 16000)
    # hm_shape = dataset.hm_shape
    hm = np.zeros((512, 512, 256))
    # print('hm_shape', hm_shape)
    # print('dataset tomo', dataset.tomos[0].shape)
    data_loader = torch.utils.data.DataLoader(PrefetchDatasetProj(opt, dataset), batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    num_iters = len(dataset)//opt.batch_size
    memory_bank_test = MemoryBank(len(dataset), 256, 2, 0.07)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot_time', 'load', 'pre', 'net', 'dec']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    all_proj_vecs = []
    # all_labels = []
    all_coords = []
    # feature_maps = []
    all_pred_vecs = []
    all_subvols = []
    all_names = []
    all_orig_subvols = []
    for iter_id, batch in enumerate(data_loader):
        input_tomo = batch['input'].to(opt.device,non_blocking=True)
        # print('input_tomo', input_tomo.shape)
        # meta = batch['meta']
        coord = batch['coord']
        name = batch['name']
        label_tomo = batch['label']
        orig_tomo = batch['orig']
        # print('coord', coord.shape)
        # ret = model.forward_test(input_tomo)
        feature = model(input_tomo)
        feature = F.normalize(feature, dim=1)
        feature = feature.detach().cpu().numpy()
        # memory_bank_test.update(ret['pred'], label_tomo)
        # if iter_id % 100 == 0:
        #     print('Fill memory bank [%d/%d]' %(iter_id, len(data_loader)))
        # # print('ret', ret)
        # proj = ret['proj'].detach().cpu().numpy()
        # pred = ret['pred'].detach().cpu().numpy()
        # all_pred_vecs.append(pred)
        all_proj_vecs.append(feature)
        all_subvols.append(input_tomo.detach().cpu().numpy())
        all_coords.append(coord)
        all_names.append(name)
        all_orig_subvols.append(orig_tomo.numpy())
        # print('proj', proj.shape)
        # meta = label_tomo
    #     ret = detector.run(input_tomo, meta)
    #     # print('ret', ret)
    #     # print('feature map', ret['fm'].shape)
    #     all_feature_vecs.append(ret['feat'])
    #     all_labels.append(ret['label'])
    #     # feature_maps.append(ret['fm'])
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   iter_id, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    # for t in avg_time_stats:
    #     avg_time_stats[t].update(ret[t])
    #     Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
    #     t, tm = avg_time_stats[t])
    bar.next()
    bar.finish()
    topk = 10
    # indices, _,_ = memory_bank_test.mine_nearest_neighbors(topk)
    # print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    # # np.save('')
    # # print(len(all_feature_vecs))
    # # print(all_feature_vecs[0].shape)
    all_proj_vecs = np.concatenate(all_proj_vecs, axis = 0)
    # all_pred_vecs = np.concatenate(all_pred_vecs, axis =0 )
    # all_labels = np.concatenate(all_labels, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)
    all_names = np.concatenate(all_names, axis=0)
    print('all_names', all_names)
    all_subvols = np.concatenate(all_subvols, axis=0)
    all_orig_subvols = np.concatenate(all_orig_subvols, axis=0)
    # print('all_subvols,', all_subvols.shape)
    print('all_coords', all_coords.shape)
    print(all_coords)
    # print('all_pred_vecs', all_pred_vecs.shape)
    # kmeans = KMeans(n_clusters = 4, random_state=0).fit(all_pred_vecs)
    # labels = kmeans.labels_
    # print(labels.shape)
    # hm_empty = np.zeros((hm_shape[0], hm_shape[1], hm_shape[2]))-1
    # print('hm_empty', hm_empty.shape)
    # for i,l in enumerate(labels):
    #     crd = all_coords[i]
    #     # print(crd)
    #     hm[crd[1],crd[0], crd[2]] = (l+1) * 10

    # with mrcfile.new('new_test_hm.mrc', overwrite=True) as mrc:
    #     mrc.set_data(np.float32(hm))
    # hm_empty -= hm_empty.min()
    # hm_empty /= hm_empty.max()
    # tilt_img = np.mean(dataset.tomos[0],axis=0)
    # tilt_img -= tilt_img.min()
    # tilt_img /- tilt_img.max()
    # merged_img = hm_empty[0] * 0.3 + tilt_img[15:512-16, 15:512-16] * 0.7
    # # tilt_img = load_rec('')
    # fig = plt.figure(figsize=(20,20))
    # plt.imshow(merged_img,cmap='gray')
    # plt.savefig('testhm.png')


    print('opt.save_dir', opt.save_dir)
    dataset_number = opt.exp_id.split('_')[0]
    # # # all_feature_maps = np.concatenate(feature_maps, axis=0)
    save_proj = os.path.join(opt.save_dir, 'preselect_{}_proj_f.npy'.format(dataset_number))
    # save_lb = os.path.join(opt.save_dir, 'shrec_labels_box32_withgold.npy')
    save_name = os.path.join(opt.save_dir, 'preselect_{}_names_f.npy'.format(dataset_number))
    # save_pred = os.path.join(opt.save_dir, 'preselect_{}_pred_f.npy'.format(dataset_number))
    save_subvols = os.path.join(opt.save_dir, 'preselect_{}_subvol_f.npy'.format(dataset_number))
    # save_indices = os.path.join(opt.save_dir, 'preselect_{}_ind_f.npy'.format(dataset_number))
    save_coords = os.path.join(opt.save_dir, 'preselect_{}_coord_f.npy'.format(dataset_number))
    save_origsub = os.path.join(opt.save_dir, 'preselect_{}_origsub_f.npy'.format(dataset_number))
    # # # all_feature_maps = np.concatenate(feature_maps, axis=0)
    # save_proj = os.path.join(opt.save_dir, 'preselect_10988_proj_f.npy')
    # # save_lb = os.path.join(opt.save_dir, 'shrec_labels_box32_withgold.npy')
    # save_name = os.path.join(opt.save_dir, 'preselect_10988_names_f.npy')
    # save_pred = os.path.join(opt.save_dir, 'preselect_10988_pred_f.npy')
    # save_subvols = os.path.join(opt.save_dir, 'preselect_10988_subvol_f.npy')
    # save_indices = os.path.join(opt.save_dir, 'preselect_10988_ind_f.npy')
    # save_coords = os.path.join(opt.save_dir, 'preselect_10988_coord_f.npy')
    # save_origsub = os.path.join(opt.save_dir, 'preselect_10988_origsub_f.npy')
    # # np.save('test_proj_vecs_box36.npy', all_proj_vecs)
    # # np.save('test_labels_box36.npy', all_labels)
    # # np.save('test_pred_vecs_box36.npy', all_pred_vecs)
    np.save(save_name, all_names)
    # np.save(save_indices, indices)
    np.save(save_subvols, all_subvols)
    # np.save(save_pred, all_pred_vecs)
    np.save(save_proj, all_proj_vecs)
    np.save(save_coords, all_coords)
    np.save(save_origsub, all_orig_subvols)
    # np.save(save_lb, all_labels)

if __name__ == '__main__':
    opt = opts().parse()
    test(opt)

