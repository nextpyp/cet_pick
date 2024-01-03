import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from cet_pick.models.loss import entropy 

@torch.no_grad()
def get_predictions_scan(opt, dataloader, model, return_features=False):
    model.eval()
    predictions = [[] for _ in range(opt.nheads)]
    probs = [[] for _ in range(opt.nheads)]
    if return_features:
        features = torch.zeros((len(dataloader.sampler), 128)).cuda()
    neighbors = []

    ptr = 0 
    for batch in dataloader:
        images = batch['anchor'].to(device = opt.device, non_blocking=True)
        bs = images.shape[0]
        res = model(images, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr:ptr+bs] = res['features']
            ptr += bs 

        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))
        neighbors.append(batch['possible_neighbors'])
    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    neighbors = torch.cat(neighbors, dim=0)
    out = [{'predictions': pred_, 'probabilities': prob_, 'neighbors': neighbors} for pred_, prob_ in zip(predictions, probs)]
    if return_features:
        return out, features.cpu()
    else:
        return out 

@torch.no_grad()
def get_predictions_scan2d3d(opt, dataloader, model, return_features=False):
    model.eval()
    predictions = [[] for _ in range(opt.nheads)]
    probs = [[] for _ in range(opt.nheads)]
    if return_features:
        features = torch.zeros((len(dataloader.sampler), 128)).cuda()
    neighbors = []

    ptr = 0 
    for batch in dataloader:
        images_2d = batch['anchor_2d'].to(device = opt.device, non_blocking=True)
        images_3d = batch['anchor_3d'].to(device = opt.device, non_blocking=True)
        bs = images_2d.shape[0]
        res = model(images_2d, images_3d, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr:ptr+bs] = res['features']
            ptr += bs 

        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))
        neighbors.append(batch['possible_neighbors'])
    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    neighbors = torch.cat(neighbors, dim=0)
    out = [{'predictions': pred_, 'probabilities': prob_, 'neighbors': neighbors} for pred_, prob_ in zip(predictions, probs)]
    if return_features:
        return out, features.cpu()
    else:
        return out 

@torch.no_grad()
def scan_evaluate(predictions):
    num_heads = len(predictions)
    output = []

    for head in predictions:
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1,1).expand_as(neighbors)
        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        # Consistency loss       
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones).item()
        
        # Total loss
        total_loss = - entropy_loss + consistency_loss
        
        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}

