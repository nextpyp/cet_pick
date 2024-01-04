#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
here = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)

import numpy as np
import pandas as pd

from cet_pick.evaluation.metrics import precision_recall_curve
from cet_pick.evaluation.algorithms import match_coordinates

name = 'precision_recall_curve'
help = 'calculate the precision-recall curve for a set of predicted particle coordinates with scores and a set of target coordinates'


def add_arguments(parser):
    parser.add_argument('--predicted', help='path to file containing predicted particle coordinates with scores')
    parser.add_argument('--targets', help='path to file specifying target particle coordinates') 
    parser.add_argument('--path', help='path to all output files')
    parser.add_argument('-r', '--assignment-radius', required=True, type=int, help='maximum distance between prediction and labeled target allowed for considering them a match')
    parser.add_argument('--images', choices=['target', 'predicted', 'union'], default='target', help='only count particles on micrographs with coordinates labeled in the targets file, the predicted file, or the union of those (default: target)')
    parser.add_argument('--stats', help='file to output max stats')
    parser.add_argument('--out', help='out file for precision recall stats')
    parser.add_argument('--folder', help='folder name')
    parser.add_argument('--w', help='overall stats file')

    return parser

def main(args):
    if os.path.exists(args.w):
        overall_output = open(args.w, 'a')
    else:
        overall_output = open(args.w, 'w')

    
    match_radius = args.assignment_radius
    # targets = pd.read_csv(args.targets, sep='\t')
    predicts = pd.read_csv(args.predicted, sep='\t', comment='#')
    # pred_path = os.path.join(args.path, args.predicted)
    targets = pd.read_csv(args.targets, sep='\t')
    # predicts = pd.read_csv(args.predicted, sep='\t', comment='#')
    # predicts = pd.read_csv(pred_path, sep='\t', comment='#')
    out_path = os.path.join(args.path, args.stats)
    # out_path = args.predicted.
    out_file = open(out_path, 'w+')
    out_prec = os.path.join(args.path, args.out)
    print('threshold' +'\t' + 'precision' + '\t' + 'recall' + '\t' + 'f1', file = out_file)
    if args.images == 'union':
        image_list = set(targets.image_name.unique()) | set(predicts.image_name.unique())
    elif args.images == 'target':
        image_list = set(targets.image_name.unique())
    elif args.images == 'predicted':
        image_list = set(predicts.image_name.unique())
    else:
        raise Exception('Unknown image argument: ' + args.images)

    image_list = list(image_list)

    N = len(targets)

    matches = []
    scores = []

    count = 0
    mae = 0
    for name in image_list:
        print('name', name)
        target = targets.loc[targets.image_name == name]
        predict = predicts.loc[predicts.image_name == name]

        target_coords = target[['x_coord', 'y_coord', 'z_coord']].values
        predict_coords = predict[['x_coord', 'y_coord', 'z_coord']].values 
        # print('predict_coords', predict_coords)
        # print('target_coords', target_coords)
        score = predict.score.values.astype(np.float32)

        match,dist = match_coordinates(target_coords, predict_coords, match_radius)
        print('match', match)
        this_mae = np.sum(dist[match==1])
        count += np.sum(match)
        delta = this_mae - np.sum(match)*mae
        mae += delta/count

        matches.append(match)
        scores.append(score)


    matches = np.concatenate(matches, 0)
    scores = np.concatenate(scores, 0)

    precision,recall,threshold,auprc = precision_recall_curve(matches, scores, N=N)
    line_stats = [args.folder, str(auprc)]
    overall_output.write('\t'.join(line_stats) + '\n')
    print('# auprc={}, mae={}'.format(auprc,np.sqrt(mae)))     

    mask = (precision + recall) == 0
    f1 = 2*precision*recall
    f1[mask] = 0
    f1[~mask] /= (precision + recall)[~mask]

    table = pd.DataFrame({'threshold': threshold})
    table['precision'] = precision
    table['recall'] = recall
    table['f1'] = f1

    table.to_csv(out_prec, sep='\t', index=False)
    max_prec = table.loc[table['precision'].idxmax()]
    max_recc = table.loc[table['recall'].idxmax()]
    max_f1 = table.loc[table['f1'].idxmax()]
    max_prec = [str(i) for i in list(max_prec)]
    max_recc = [str(i) for i in list(max_recc)]
    max_f1 = [str(i) for i in list(max_f1)]
    # print('max_prec', max_prec)
    out_file.write('\t'.join(max_prec) + '\n')
    out_file.write('\t'.join(max_recc) + '\n')
    out_file.write('\t'.join(max_f1) + '\n')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for calculating the precision-recall curve for a set of predicted particle coordinates and a set of target coordinates.')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


