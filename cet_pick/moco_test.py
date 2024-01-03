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
from cet_pick.datasets.dataset_factory import dataset_factory
from cet_pick.detectors.detector_factory import detector_factory
import torchio as tio 
from cet_pick.utils.loader import load_tomos_from_list, cutup
from cet_pick.models.model import create_model, load_model 
from cet_pick.utils.memory_bank import MemoryBank

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
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()
    print('model', model)
    # detector = Detector(opt)
    # split = 'val' if not opt.trainval else 'test'
    split = 'test'

    dataset = Dataset(opt, split,(8,64,64))
    data_loader = torch.utils.data.DataLoader(PrefetchDataset(opt, dataset), batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    num_iters = len(dataset)//opt.batch_size
    memory_bank_test = MemoryBank(len(dataset), 128, 2, 0.07)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot_time', 'load', 'pre', 'net', 'dec']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    all_proj_vecs = []
    all_labels = []
    # feature_maps = []
    all_pred_vecs = []
    all_subvols = []
    for iter_id, batch in enumerate(data_loader):
        input_tomo = batch['input'].to(opt.device)
        print('input_tomo', input_tomo.shape)
        # meta = batch['meta']
        label_tomo = batch['label']
        ret = model.forward_test(input_tomo)
        memory_bank_test.update(ret['proj'], label_tomo)
        if iter_id % 100 == 0:
            print('Fill memory bank [%d/%d]' %(iter_id, len(data_loader)))
        # print('ret', ret)
        proj = ret['proj'].detach().cpu().numpy()
        # pred = ret['pred'].detach().cpu().numpy()
        # all_pred_vecs.append(pred)
        all_proj_vecs.append(proj)
        all_subvols.append(input_tomo.detach().cpu().numpy())
        all_labels.append(label_tomo)
        print('proj', proj.shape)
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
    topk = 20
    indices, acc = memory_bank_test.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    # np.save('')
    # print(len(all_feature_vecs))
    # print(all_feature_vecs[0].shape)
    all_proj_vecs = np.concatenate(all_proj_vecs, axis = 0)
    # all_pred_vecs = np.concatenate(all_pred_vecs, axis =0 )
    all_labels = np.concatenate(all_labels, axis=0)
    all_subvols = np.concatenate(all_subvols, axis=0)
    print('all_subvols,', all_subvols.shape)
    # # all_feature_maps = np.concatenate(feature_maps, axis=0)
    save_proj = os.path.join(opt.save_dir, 'shrec_proj_vecs_box32_withgold.npy')
    save_lb = os.path.join(opt.save_dir, 'shrec_labels_box32_withgold.npy')
    # save_pred = os.path.join(opt.save_dir, 'shrec_pred_vecs_box32_withgold.npy')
    save_subvols = os.path.join(opt.save_dir, 'shrec_subvols_box32_withgold.npy')
    save_indices = os.path.join(opt.save_dir, 'shrec_knn_indicies_box32_withgold.npy')
    # np.save('test_proj_vecs_box36.npy', all_proj_vecs)
    # np.save('test_labels_box36.npy', all_labels)
    # np.save('test_pred_vecs_box36.npy', all_pred_vecs)
    np.save(save_indices, indices)
    np.save(save_subvols, all_subvols)
    # np.save(save_pred, all_pred_vecs)
    np.save(save_proj, all_proj_vecs)
    np.save(save_lb, all_labels)

if __name__ == '__main__':
    opt = opts().parse()
    test(opt)

