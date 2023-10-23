from __future__ import absolute_import, print_function, division
import numpy as np 
from scipy import spatial
import os 
import glob

def add_arguments(parser):
    parser.add_argument('--path', help='path to file containing predicted particle coordinates with scores')
    parser.add_argument('--gold', help='path to file specifying target particle coordinates') 
    parser.add_argument('--r', help='radius as closest gold')
    parser.add_argument('--out', help='out filtered txt path')
    return parser 

def main(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    all_ours = os.path.join(args.path,'TS*.txt')
    existing_outs = glob.glob(all_ours)
    trail = '_gold3d.txt'
    for p in existing_outs:
        coord_a = p
        # print(p)
        name = p.split('/')[-1].split('.')[0]
        print(name)
        gold_name = name + trail
        coord_gold = os.path.join(args.gold, gold_name)
        print(coord_gold)
        coords_ours = []
        with open(coord_a, 'r') as f:
            for l in f:
                l = l.split()
                l = [float(i) for i in l]
        #         print(l)
                coords_ours.append(l)
        coords_gold = []
        with open(coord_gold, 'r') as f:
            for l in f:
                l = l.split()
                l = [float(i) for i in l]
        #         print(l)
                coords_gold.append(l)
        coords_ours = np.array(coords_ours)
        coords_gold = np.array(coords_gold)
        dist = spatial.distance.cdist(coords_ours, coords_gold)
        keep_coords = []
        min_dist = np.min(dist,axis=1)
        for i, j in enumerate(min_dist):
            if j > 20:
                keep_coords.append(coords_ours[i])
        out_name = name + '.txt'
        out_path = os.path.join(args.out, out_name)
        with open(out_path, 'w+') as f1:
            for l in keep_coords:
                l = [str(int(i)) for i in l]
                f1.write('\t'.join(l) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for remove gold detections.')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)