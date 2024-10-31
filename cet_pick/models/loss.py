from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from cet_pick.models.utils import _transpose_and_gather_feat
import numpy as np
from scipy import stats
from torch.autograd import Variable

EPS=1e-8

class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling
        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss
        
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)

def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)
    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        # total_loss = consistency_loss
        
        return total_loss, consistency_loss, entropy_loss


class SupConLossPre(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossPre, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # print('features', features.shape)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def _bce_loss(pred, gt):
    bce_loss = nn.BCEWithLogitsLoss()

    loss = bce_loss(pred, gt)
    return loss 

def _pu_ge_loss(pred, gt, tau, criteria, slack=1, entropy_penalty=0):
    pred = pred.squeeze()
    gt = gt.squeeze()
    gt = gt.view(-1)
    pred = pred.view(-1)
    select = (gt.data >= 0)

    if select.sum().item() >= 0:
        classifier_loss = criteria(pred[select], gt[select])
    else:
        classifier_loss = 0  

    select = (gt.data == -1)

    N = select.sum().item()
    # p_hat = torch.sigmoid(out_score[select])
    p_hat = pred[select]
    # p_hat = torch.sigmoid(pred[select])
    q_mu = p_hat.sum()
    q_var = torch.sum(p_hat*(1-p_hat))
    count_vector = torch.arange(0, N+1).float()
    count_vector = count_vector.to(q_mu.device)
    q_discrete = -0.5*(q_mu - count_vector)**2/(q_var + 1e-7)
    q_discrete = F.softmax(q_discrete, dim=0)

    log_binom = stats.binom.logpmf(np.arange(0, N+1), N, tau)
    log_binom = torch.from_numpy(log_binom).float()
    if q_var.is_cuda:
        log_binom = log_binom.cuda()
    log_binom = Variable(log_binom)
    # ge_penalty = -torch.mean(log_binom*q_discrete)
    ge_penalty = -torch.sum(log_binom*q_discrete)
    if entropy_penalty > 0:
        q_entropy = 0.5 * (torch.log(q_var) + np.log(2*np.pi) + 1)
        ge_penalty = ge_penalty + q_entropy * entropy_penalty
    loss = classifier_loss + slack*ge_penalty
    # loss = loss.mean()

    return loss 

def _pu_neg_loss(pred, gt, tau, beta, gamma):
    '''
    Positive Unlabeled Focal Loss 
    Arguments:
    pred (batch x c x h x w)
    gt (batch x c x h x w)
    
    '''
    # gt = gt.unsqueeze(0)
    pred = pred.squeeze()
    gt = gt.squeeze()
    true_pos_inds = gt.eq(1).float()
    other_inds = gt.lt(1).float()
    labeled_inds = gt.gt(-1).float()
    soft_pos_inds = labeled_inds == other_inds
    soft_pos_inds = soft_pos_inds.float()
    unlabeled_inds = gt.eq(-1).float()

    # num_pos = true_pos_inds.float().sum() + soft_pos_inds.float().sum()
    num_pos = true_pos_inds.float().sum()
    if num_pos == 0:
        raise ValueError('Num of true positive is zero, please check tomogram size of input coordinates order or use smaller translation ratio')
    num_unlabeld = unlabeled_inds.float().sum()
    num_soft = soft_pos_inds.float().sum()
    # total_pos = (num_pos+num_soft)/(num_pos+num_unlabeld+num_soft)
    # tau_use = tau - total_pos/2
    soft_pow_weights = torch.pow(1 - gt, 4)
    soft_pow_neg_weights = torch.pow(gt, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * true_pos_inds
    if num_soft > 0:
        soft_pos_loss = torch.log(1 - pred) * torch.pow(pred, 2) * soft_pow_weights * soft_pos_inds
        pos_loss_tot = -(pos_loss.sum())/ num_pos - (soft_pos_loss.sum())/num_soft
    else: 
        pos_loss_tot = -(pos_loss.sum())/ num_pos
    # pos_loss_tot = -(pos_loss + soft_pos_loss).sum()
    pos_risk = (pos_loss_tot) * tau 
    neg_pos_loss = torch.log(1-pred) * torch.pow(pred, 2) * true_pos_inds
    if num_soft > 0:
        neg_soft_pos_loss = torch.log(pred) * torch.pow(1-pred, 2) * soft_pow_neg_weights * soft_pos_inds
    # neg_pos_loss_tot = -(neg_pos_loss + neg_soft_pos_loss).sum()
        neg_pos_loss_tot = -(neg_pos_loss.sum()) / num_pos - (neg_soft_pos_loss.sum())/num_soft
    else: 
        neg_pos_loss_tot = -(neg_pos_loss.sum()) / num_pos
    neg_pos_risk = neg_pos_loss_tot 
    unlabeled_neg_loss = torch.pow(pred, 2) * torch.log(1 - pred) * unlabeled_inds
    unlabeled_loss = -(unlabeled_neg_loss).sum()
    unlabeled_risk = unlabeled_loss / num_unlabeld

    neg_risk_total = -tau * neg_pos_risk + unlabeled_risk

    if neg_risk_total < -beta:
        return pos_risk
    else:
        return pos_risk + neg_risk_total

class PULoss(nn.Module):

    """
    Non-negative voxel-level positive unlabeled loss 
    """

    def __init__(self, tau, beta = 0, gamma = 1):
        super(PULoss, self).__init__()
        self.tau = tau 

        self.gamma = gamma 
        self.beta = beta 
        self.puloss = _pu_neg_loss

    def forward(self, pred, gt):
        return self.puloss(pred, gt, self.tau, self.beta, self.gamma)

class PUGELoss(nn.Module):
    def __init__(self, tau, criteria, slack=1, entropy_penalty=0):
        super(PUGELoss, self).__init__()
        self.tau = tau 
        self.slack = slack 
        self.entropy_penalty = entropy_penalty 
        self.puloss = _pu_ge_loss
        self.criteria = criteria

    def forward(self, pred, gt):
        return self.puloss(pred, gt, self.tau, self.criteria, self.slack, self.entropy_penalty)

def _neg_loss_mod(pred, gt, threshold):
    ''' Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
    pred (batch x c x h x w)
    gt_regr (batch x c x h x w)
    '''
    

    gt = gt.unsqueeze(0)
    # pos_inds = gt.eq(0.9).float()
    pos_inds = gt.gt(threshold).float()
    # pos_inds = gt.gt(threshold).float()
    neg_inds_update = gt.lt(threshold).float()

    # print('gt', gt)
    # gt_0 = gt.gt(-1).float()
    # neg_inds_update = gt_0 == neg_inds
    # neg_inds_update = neg_inds_update.float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds_update

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
    pred (batch x c x h x w)
    gt_regr (batch x c x h x w)
    '''
    

    gt = gt.unsqueeze(0)
    # pos_inds = gt.eq(0.9).float()
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    gt_0 = gt.gt(-1).float()
    neg_inds_update = gt_0 == neg_inds
    neg_inds_update = neg_inds_update.float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds_update

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    
    return loss

def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class FocalLoss_mod(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self, threshold):
        super(FocalLoss_mod, self).__init__()
        self.neg_loss = _neg_loss_mod
        self.threshold = threshold

    def forward(self, out, target):
        return self.neg_loss(out, target, self.threshold)

class RegLoss(nn.Module):
    def __init__(self):
        """
        Regression loss for an output tensor
        output (batch * c * d * h * w)
        mask (batch  * max_objects)
        ind (batch * max_objects)
        target (batch * max_objects * dim)
        """
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss 

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, gt):

        gt = gt.flatten()
        pred = pred.flatten()
        loss = self.bce(pred, gt)
        return loss

class BiasedConLoss(nn.Module):
    """
    Contrastive regularization without debiasing term 

    """
    def __init__(self, base_temperature):
        super(BiasedConLoss, self).__init__()
        self.base_temperature = base_temperature 
    def forward(self, labels, all_features, all_features_cr, opt):
        # features of dimension num_of_pixels * dim 
        # labels of dimension num_of_pixels 
        # number of positives = number of labels = particle 

        labels_postivies = labels.eq(1).float()
        # labels_postivies = labels.gt(0.8).float()
        # unlabels = labels.lt(1)
        num_of_positives = labels_postivies.sum()
        num_of_pixels = all_features.shape[0]
        num_of_negatives = 2 * (num_of_pixels - num_of_positives)
        self_mask = torch.zeros((num_of_pixels*2, num_of_pixels*2))
        self_mask[:num_of_pixels,num_of_pixels:] = torch.eye(num_of_pixels)
        self_mask[num_of_pixels:, :num_of_pixels] = torch.eye(num_of_pixels)
        self_mask = self_mask.to(opt.device)

        del labels_postivies
        out_total = torch.cat([all_features, all_features_cr], dim=0)
        del all_features, all_features_cr
        out_sims = torch.mm(out_total, out_total.t().contiguous()) / self.base_temperature
        mask = torch.eye(out_sims.shape[0]).to(opt.device)
        mask = 1 - mask
        del out_total
        torch.cuda.empty_cache()
        logits_max_all, _ = torch.max(out_sims, dim=1, keepdim=True)
        out_sims = out_sims - logits_max_all.detach()
        out_sims = out_sims * mask
        out_sims = torch.exp(out_sims)
        all_labels = torch.cat([labels, labels], dim=0)
        # all_out_prob = torch.cat([out_labels, out_labels],dim=0)
        # all_out_preds = torch.cat([out_labels, out_labels_cr], dim=0)
        pos_labels = all_labels.eq(1).float().bool()

        # pos_labels = all_labels.gt(0.2).float().bool()
        # un_labels = all_labels.lt(0).float().bool()
        neg_labels = all_labels.lt(1).float().bool()
        pos_features = out_sims[pos_labels,:]
        unlabeled_features = out_sims[neg_labels,:]
        unlabeled_mask = self_mask[neg_labels,:]
        del out_sims
        pos_labels = pos_labels.float()
        other_inds = all_labels.lt(1).float()
        # other_inds = all_labels.lt(0.2).float()
        labeled_inds = all_labels.gt(-1).float()
        soft_pos_inds = labeled_inds == other_inds
        neg_labels = soft_pos_inds.float()
        # neg_labels = 1 - pos_labels
        # neg_labels_all = all_labels.gt(0).float()
        num_of_other_inds = other_inds.sum()
        # pos_feat_mean = (pos_features * pos_labels).sum(dim=1)/(pos_labels.sum(0)-1)
        pos_num = torch.log(pos_features) * pos_labels
        pos_loss = pos_num - torch.log(pos_features.sum(1, keepdim=True))
        pos_loss_mean = -pos_loss.sum(1) / (pos_labels.sum())
        # rem_feat_mean = (pos_features * other_inds).sum(dim=1)/(other_inds.sum(0))
        # Ng = self.calc_g(pos_feat_mean, rem_feat_mean, self.tau_plus) 
        # debiased_loss_sup = -torch.log((pos_feat_mean)/ ((pos_feat_mean) + Ng))
        # del Ng 
        # del pos_features 
        # rem_labels = 1 - unlabeled_mask
        unlabed_pos_feat_mean = torch.log(unlabeled_features) * unlabeled_mask
        unlabeled_loss = unlabed_pos_feat_mean - torch.log(unlabeled_features.sum(1, keepdim=True))
        unlabeled_loss_mean = -unlabeled_loss.sum(1) / (other_inds.sum())
        # unlabeled_rem_feat_mean = (unlabeled_features * rem_labels).sum(dim=1)/num_of_negatives
        # Ng_pos = self.calc_g(unlabed_pos_feat_mean, unlabeled_rem_feat_mean, self.tau_plus) 
        # Ng_neg = self.calc_g(unlabed_pos_feat_mean, unlabeled_rem_feat_mean, 1-self.tau_plus) 
        # unlbed_prob = all_out_preds[un_labels]
        # debiased_loss_unsup = -torch.log(unlabed_pos_feat_mean / (unlabed_pos_feat_mean + Ng_pos)) * unlbed_prob - torch.log(unlabed_pos_feat_mean/(unlabed_pos_feat_mean + Ng_neg)) * (1-unlbed_prob)
        # debiased_loss_unsup = -torch.log(unlabed_pos_feat_mean / (unlabed_pos_feat_mean + Ng_pos))
        biased_loss_sup = pos_loss_mean.mean()
        biased_loss_unsup = unlabeled_loss_mean.mean()
        # debiased_loss_sup = debiased_loss_sup.mean()
        # debiased_loss_unsup = debiased_loss_unsup.mean()

        return biased_loss_sup, biased_loss_unsup

class UnbiasedConLoss(nn.Module):
    """
    Debiased contrastive regularization 

    """
    def __init__(self, base_temperature, class_prob):
        """
        arguments:
        base_temperature: temperature for infoNCE loss
        class_prob: class prior probability for positive class

        """
        super(UnbiasedConLoss, self).__init__()
        self.base_temperature = base_temperature 

        self.tau_plus = class_prob

    def calc_g(self, pos_mean, neg_mean, class_prob):
        Ng = (neg_mean - class_prob * pos_mean ) / (1-class_prob)
        Ng = torch.clamp(Ng, min = np.e**(-1 / self.base_temperature))

        return Ng

    def forward(self, labels, out_labels, out_labels_cr, all_features, all_features_cr, opt):
        # features of dimension num_of_pixels * dim 
        # labels of dimension num_of_pixels 
        # number of positives = number of labels = particle 

        # positive label indices - this can be changed to different threshold 
        # labels_postivies = labels.eq(1).float()
        if opt.thresh < 1:
            labels_postivies = labels.gt(opt.thresh).float()
        else:
            labels_postivies = labels.eq(1).float()

        num_of_positives = labels_postivies.sum()
        if num_of_positives == 0:
            raise ValueError('Num of true positive is zero, please check tomogram size of input coordinates order or use smaller translation ratio')
        num_of_pixels = all_features.shape[0]
        pos_ratio = num_of_positives / num_of_pixels
        num_of_negatives = 2 * (num_of_pixels - num_of_positives)
        self_mask = torch.zeros((num_of_pixels*2, num_of_pixels*2))
        self_mask[:num_of_pixels,num_of_pixels:] = torch.eye(num_of_pixels)
        self_mask[num_of_pixels:, :num_of_pixels] = torch.eye(num_of_pixels)
        self_mask = self_mask.to(opt.device)

        del labels_postivies
        out_total = torch.cat([all_features, all_features_cr], dim=0)
        del all_features, all_features_cr
        

        #calculate cosine similarity 
        out_sims = torch.mm(out_total, out_total.t().contiguous()) / self.base_temperature
        mask = torch.eye(out_sims.shape[0]).to(opt.device)
        mask = 1 - mask
        del out_total
        torch.cuda.empty_cache()
        logits_max_all, _ = torch.max(out_sims, dim=1, keepdim=True)
        out_sims = out_sims - logits_max_all.detach()
        out_sims = out_sims * mask
        out_sims = torch.exp(out_sims)
        all_labels = torch.cat([labels, labels], dim=0)
        # all_out_prob = torch.cat([out_labels, out_labels],dim=0)
        all_out_preds = torch.cat([out_labels, out_labels_cr], dim=0)
        if opt.thresh < 1:
            pos_labels = all_labels.gt(opt.thresh).float().bool()
        else:
            pos_labels = all_labels.eq(1).float().bool()
        # pos_labels = all_labels.gt(0.2).float().bool()
        un_labels = all_labels.lt(0).float().bool()
        pos_features = out_sims[pos_labels,:]
        unlabeled_features = out_sims[un_labels,:]
        unlabeled_mask = self_mask[un_labels,:]
        del out_sims
        pos_labels = pos_labels.float()
        other_inds = all_labels.lt(opt.thresh).float()
        labeled_inds = all_labels.gt(-1).float()
        soft_pos_inds = labeled_inds == other_inds
        neg_labels = soft_pos_inds.float()

        num_of_other_inds = other_inds.sum()
        pos_feat_mean = (pos_features * pos_labels).sum(dim=1)/(pos_labels.sum(0)-1)
        # pos_feat_mean = (pos_features * pos_labels).sum(dim=1)
        # rem_feat_mean = (pos_features * other_inds).sum(dim=1)
        rem_feat_mean = (pos_features * other_inds).sum(dim=1)/(other_inds.sum(0))
        Ng = self.calc_g(pos_feat_mean, rem_feat_mean, self.tau_plus) 
        debiased_loss_sup = -torch.log((pos_feat_mean)/ ((pos_feat_mean) + Ng))
        del Ng 
        del pos_features 
        rem_labels = 1 - unlabeled_mask
        unlabed_pos_feat_mean = (unlabeled_features * unlabeled_mask).sum(dim = 1)
        unlabeled_rem_feat_mean = (unlabeled_features * rem_labels).sum(dim=1)/num_of_negatives
        Ng_pos = self.calc_g(unlabed_pos_feat_mean, unlabeled_rem_feat_mean, self.tau_plus) 

        Ng_neg = self.calc_g(unlabed_pos_feat_mean, unlabeled_rem_feat_mean, 1-self.tau_plus) 
        unlbed_prob = all_out_preds[un_labels]
        unlbed_prob_pos = unlbed_prob.gt(0.99).float()
        mid_pos_up = unlbed_prob.lt(0.99)
        mid_pos_bot = unlbed_prob.gt(0.01)
        mid_pos_fin = mid_pos_up == mid_pos_bot
        mid_pos_fin = mid_pos_fin.bool()
        num_of_pseudopos = unlbed_prob_pos.sum()
        unlbed_prob_neg = unlbed_prob.lt(0.01).float()
        num_of_pseudonegs = unlbed_prob_neg.sum()
        num_of_mids = mid_pos_fin.float().sum()
        debiased_loss_unsup = 0
        if num_of_pseudopos > 0:
            unlabeled_pos_loss = (-torch.log(unlabed_pos_feat_mean / (unlabed_pos_feat_mean + Ng_pos)) * unlbed_prob)[unlbed_prob.gt(0.99).bool()].mean()
            debiased_loss_unsup += unlabeled_pos_loss
        if num_of_pseudonegs > 0:
            unlabeled_neg_loss = (-torch.log(unlabed_pos_feat_mean / (unlabed_pos_feat_mean + Ng_neg)) * (1-unlbed_prob))[unlbed_prob.lt(0.01).bool()].mean()
            debiased_loss_unsup += unlabeled_neg_loss
        if num_of_mids > 0:
            debiased_loss_rem = (-torch.log(unlabed_pos_feat_mean / (unlabed_pos_feat_mean + Ng_pos)) * unlbed_prob)[mid_pos_fin].mean() - (torch.log(unlabed_pos_feat_mean/(unlabed_pos_feat_mean + Ng_neg)) * (1-unlbed_prob))[mid_pos_fin].mean()
        else:
            debiased_loss_rem = 0
        # debiased contrastive loss 
        # debiased_loss_unsup = -torch.log(unlabed_pos_feat_mean / (unlabed_pos_feat_mean + Ng_pos)) * unlbed_prob - torch.log(unlabed_pos_feat_mean/(unlabed_pos_feat_mean + Ng_neg)) * (1-unlbed_prob)
        # debiased_loss_unsup = -torch.log(unlabed_pos_feat_mean / (unlabed_pos_feat_mean + Ng_pos))
        if debiased_loss_unsup != 0 and debiased_loss_rem != 0:
            debiased_loss_unsup = debiased_loss_unsup.mean() + debiased_loss_rem
        elif debiased_loss_unsup == 0 and debiased_loss_rem != 0:
            debiased_loss_unsup = debiased_loss_rem
        elif debiased_loss_unsup !=0 and debiased_loss_rem == 0:
            debiased_loss_unsup = debiased_loss_unsup.mean()
        debiased_loss_sup = debiased_loss_sup.mean()


        return debiased_loss_sup, debiased_loss_unsup

class ConsistencyLoss(nn.Module):
    """
    Loss for consistency regularization 

    """
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, out_prob, out_prob_cr):

        mse = self.mse(out_prob, out_prob_cr)
        return mse




class UnSupConLoss(nn.Module):
    def __init__(self, base_temperature):
        super(UnSupConLoss, self).__init__()
        self.base_temperature = base_temperature

    def forward(self, anc_features, all_features, probs, used_mask, num_of_pos, opt, negs = True):
        all_features = all_features.view(1, 16, -1)
        all_features = all_features.squeeze()

        feat_dot_anchor = torch.matmul(all_features.T, anc_features.T)
        del all_features
        torch.cuda.empty_cache()
        feat_dot_anchor = torch.div(feat_dot_anchor.cpu(), self.base_temperature)
        used_mask = used_mask.bool()
        logits_max_all, _ = torch.max(feat_dot_anchor, dim=1, keepdim=True)
        # feat_dot_anchor = feat_dot_anchor - logits_max_all.detach()
        labels_positives = torch.zeros(feat_dot_anchor.shape)
        if not negs:
            labels_positives[:,:num_of_pos] = 1
        denominator_ancs = torch.exp(feat_dot_anchor)
        probs = probs.view(1, 1, -1)
        probs = probs.squeeze(1)
        probs = probs.T # so it is of shape #of pos * 1

        numerator_scale = feat_dot_anchor * probs.cpu()
        del feat_dot_anchor

        log_prob_pos = numerator_scale  - torch.log(denominator_ancs.sum(1, keepdim=True))

        mean_log_prob = (labels_positives * log_prob_pos).sum(1) / (labels_positives.sum(1))

        del log_prob_pos
        del denominator_ancs
        del labels_positives 
        del numerator_scale
        mean_log_prob = mean_log_prob.to(opt.device)
        used_mask = used_mask.view(-1)
        actual_used = mean_log_prob[~used_mask]
        total_loss = -actual_used.mean()

        return total_loss

class SupConLossV2_more(nn.Module):
    def __init__(self, base_temperature, contrast_mode = 'all'):
        super(SupConLossV2_more, self).__init__()
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(self, labels, out_labels, out_labels_cr, all_features, all_features_cr, opt):
        # features of shape 1 * 16 * number of areas 

        labels_positives = labels.gt(opt.thresh).float()
        num_of_positives = labels_positives.sum()
        num_of_pixels = all_features.shape[0]
        num_of_negatives = 2 * (num_of_pixels - num_of_positives)
        self_mask = torch.zeros((num_of_pixels*2, num_of_pixels*2))
        self_mask[:num_of_pixels,num_of_pixels:] = torch.eye(num_of_pixels)
        self_mask[num_of_pixels:, :num_of_pixels] = torch.eye(num_of_pixels)
        self_mask = self_mask.to(opt.device)

        del labels_positives
        out_total = torch.cat([all_features, all_features_cr], dim=0)
        del all_features, all_features_cr
        out_sims = torch.mm(out_total, out_total.t().contiguous()) / self.base_temperature
        mask = torch.eye(out_sims.shape[0]).to(opt.device)
        mask = 1 - mask
        del out_total
        torch.cuda.empty_cache()
        logits_max_all, _ = torch.max(out_sims, dim=1, keepdim=True)
        out_sims = out_sims - logits_max_all.detach()
        out_sims = out_sims * mask
        out_sims = torch.exp(out_sims)
        all_labels = torch.cat([labels, labels], dim=0)
        # if we have cr, use this contrastive addition 
    
        # pos_labels = all_labels.eq(1).float().bool()
        pos_labels = all_labels.gt(opt.thresh).float().bool()
        un_labels = all_labels.lt(opt.thresh).float().bool()
        pos_features = out_sims[pos_labels,:]
        unlabeled_features = out_sims[un_labels,:]
        unlabeled_mask = self_mask[un_labels,:]
        del out_sims
        pos_labels = pos_labels.float()
        # other_inds = all_labels.lt().float()
        # other_inds = all_labels.lt(0.2).float()
        # labeled_inds = all_labels.gt(-1).float()
        # soft_pos_inds = labeled_inds == other_inds
        # neg_labels = soft_pos_inds.float()
        neg_labels = un_labels.float()
        log_prob_pos = torch.log(pos_features) - torch.log(pos_features.sum(1, keepdim=True))
        log_prob_neg = torch.log(unlabeled_features) - torch.log(unlabeled_features.sum(1, keepdim=True))
        mean_log_prob_pos = (log_prob_pos * pos_labels).sum(1)/(pos_labels.sum(0))
        mean_log_prob_negs = (log_prob_neg * unlabeled_mask).sum(1)
        # mean_log_prob_negs = (log_prob_neg[:,hm_neg_bool]).sum(1)/hm_neg.sum(0)
        # mean_log_prob_negs = mean_log_prob_negs.to(opt.device)

        # del denominator_soft
       
        # total_loss = -mean_log_prob_pos.mean()
        total_loss = -mean_log_prob_pos.mean()  - mean_log_prob_negs.mean()
        # total_loss = -mean_log_prob_pos.mean()
        return total_loss


class SupConLossV2(nn.Module):
    def __init__(self, base_temperature, temperature_soft, temperature_hard, contrast_mode = 'all'):
        super(SupConLossV2, self).__init__()
        self.temperature_soft = temperature_soft
        self.temperature_hard = temperature_hard
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(self, all_features, hm, opt):
        # features of shape 1 * 16 * number of areas 
        all_features = all_features.reshape((1, 16, -1))
        all_features = all_features.squeeze()

        cos_sim = torch.matmul(all_features.T, all_features)
        mask = torch.eye(cos_sim.shape[0]).to(opt.device)
        mask = 1 - mask 
        cos_sim = cos_sim * mask
        hm_flatten = hm.view(1, 1, -1)
        hm_flatten = hm_flatten.squeeze()
        hm_pos = hm_flatten.gt(opt.thresh).float()
        hm_pos_bool = hm_pos.bool()
        # negative 
        hm_neg = hm_flatten.lt(opt.thresh).float()
        hm_neg_bool = hm_neg.bool()
        del hm_flatten
        pos_features = cos_sim[hm_pos_bool, :]
        neg_features = cos_sim[hm_neg_bool, :]
        pos_features = torch.div(pos_features, self.base_temperature)
        neg_features = torch.div(neg_features, self.base_temperature)
        logits_max_pos, _ = torch.max(pos_features, dim=1, keepdim=True)
        logits_max_neg, _ = torch.max(neg_features, dim=1, keepdim=True)
        pos_features = pos_features - logits_max_pos.detach()
        neg_features = neg_features - logits_max_neg.detach()
        del cos_sim
        del logits_max_pos
        del logits_max_neg
        denominator_pos = torch.exp(pos_features)
        denominator_neg = torch.exp(neg_features)

        log_prob_pos = pos_features - torch.log(denominator_pos.sum(1, keepdim=True))
        log_prob_neg = neg_features - torch.log(denominator_neg.sum(1, keepdim=True))
        del denominator_neg
        del denominator_pos
        mean_log_prob_pos = (log_prob_pos * hm_pos).sum(1)/(hm_pos.sum(0))
        mean_log_prob_negs = (log_prob_neg * hm_neg).sum(1)/(hm_neg.sum(0))

        total_loss = -mean_log_prob_pos.mean()  - mean_log_prob_negs.mean()
        return total_loss


class KMeansVMFLoss(nn.Module):
    def __init__(self, temp):
        super(KMeansVMFLoss, self).__init__()
        self.temp = temp

    def _cosine_similarity(self, u, v):

        # num_of_samples * hidden_dim
        u = F.normalize(u, dim = 1)
        v = F.normalize(v, dim = 1)

        similarity = torch.div(torch.matmul(u, v.T), self.temp)
        logits_max_pos, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - logits_max_pos.detach()


        similarity = torch.exp(similarity)

        return similarity

    def forward(self, embeddings, labels, prototypes, opt):
        similarities = self._cosine_similarity(embeddings, prototypes)
        num_of_pixels = embeddings.shape[0]

        # get similarities of #number_of_pixels * num_of_prototypes 
        # labels of dim #num of pixels * 1
        labels = labels.squeeze().long()
        one_hot_labels = F.one_hot(labels).to(device=opt.device)
        opposite_mask = 1 - one_hot_labels
        numerator = torch.sum(similarities * one_hot_labels, axis = 1)
        denominator = torch.sum(similarities, axis = 1)
        prob = torch.div(numerator, denominator)
        prob = torch.sum(torch.log(prob)) * (-1/num_of_pixels)
        return prob

class PartialSupLoss(nn.Module):
    def __init__(self, temp):
        super(PartialSupLoss, self).__init__()
        self.temp = temp 

    def forward(self, embeddings, gt_labels, opt):
        gt_labels_f = gt_labels.squeeze()
        labeled_embeddings = embeddings[gt_labels_f > 0]
        gt_labels = gt_labels[gt_labels > 0]
        cos_sims = torch.matmul(labeled_embeddings, labeled_embeddings.T)
        cos_sims = torch.div(cos_sims, self.temp)
        mask = torch.eye(cos_sims.shape[0]).to(opt.device)
        mask = 1 - mask
        mask = mask.to(device=opt.device)
        gt_labels = gt_labels.contiguous().view(-1, 1)
        lb_mask = torch.eq(gt_labels, gt_labels.T).float()
        lb_mask = lb_mask - torch.eye(mask.shape[0], device= opt.device)
        logits_max, _ = torch.max(cos_sims, dim=1, keepdim=True)
        logits = cos_sims - logits_max.detach()

        logits_numer = logits * lb_mask
        logits_denom = torch.exp(logits) * mask
        log_prob = logits_numer - torch.log(logits_denom.sum(1, keepdim=True))
        mean_log_prob = log_prob.sum(1) / lb_mask.sum(1)
        loss = -1 * mean_log_prob 
        loss = loss.mean()

        return loss


        
class SupConLoss(nn.Module):
    def __init__(self, base_temperature, temperature_soft, temperature_hard, contrast_mode = 'all'):
        super(SupConLoss, self).__init__()
        self.temperature_soft = temperature_soft
        self.temperature_hard = temperature_hard
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(self, pos_features, soft_negs, opt, labels=None):
        anchor_count = pos_features.shape[0]

        all_features = torch.cat((pos_features, soft_negs), dim=0)
        anchor_dot_contrast = torch.matmul(all_features, all_features.T)
        # mask out diagonal 
        mask = torch.eye(all_features.shape[0]).to(opt.device)
        mask = mask.bool()
        anchor_dot_contrast = anchor_dot_contrast[~mask].view(all_features.shape[0], -1)
        anchor_dot_contrast_pos = torch.div(anchor_dot_contrast[:pos_features.shape[0]], self.base_temperature)
        # del anchor_dot_contrast
        anchor_dot_contrast_soft = torch.div(anchor_dot_contrast[pos_features.shape[0]:], self.temperature_soft)
        # anchor_dot_contrast_neg = torch.div(anchor_dot_contrast[-hard_negs.shape[0]:], self.temperature_hard)
        del anchor_dot_contrast
        logits_max_pos, _ = torch.max(anchor_dot_contrast_pos, dim=1, keepdim=True)
        logits_max_soft, _ = torch.max(anchor_dot_contrast_soft, dim=1, keepdim=True)
        num_of_positives = pos_features.shape[0]
        num_of_hard_negs = soft_negs.shape[0]
        labels_positives = torch.zeros(anchor_dot_contrast_pos.shape).to(opt.device)
        labels_soft = torch.zeros(anchor_dot_contrast_soft.shape).to(opt.device)
        labels_positives[:,:(anchor_dot_contrast_pos.shape[0]-1)] = 1
        labels_soft[:,(anchor_dot_contrast_pos.shape[0]-1):] = 1
        denominator_pos = torch.exp(anchor_dot_contrast_pos)
        denominator_soft = torch.exp(anchor_dot_contrast_soft)
        log_prob_pos = anchor_dot_contrast_pos - torch.log(denominator_pos.sum(1, keepdim=True))
        log_prob_soft = anchor_dot_contrast_soft - torch.log(denominator_soft.sum(1,keepdim=True))
        mean_log_prob_pos = (labels_positives * log_prob_pos).sum(1)/(labels_positives.sum(1))
        mean_log_prob_soft = (labels_soft * log_prob_soft).sum(1)/(labels_soft.sum(1))
        del anchor_dot_contrast_pos
        del anchor_dot_contrast_soft
        del log_prob_soft
        del log_prob_pos 
        del denominator_soft
        del denominator_pos
        del labels_positives 
        del labels_soft
        total_loss = -mean_log_prob_pos.mean() - mean_log_prob_soft.mean()

        return total_loss




