import os
import sys

import numpy as np  
import pandas as pd 
import glob

def add_arguments(parser):
    parser.add_argument('--path', help='path to file containing predicted particle coordinates with scores')
    
    parser.add_argument('--out', help='output file for all merged coordinates')
    return parser

def main(args):
	path_to_out = args.path 
	all_txts = os.path.join(path_to_out, '*.txt')
	txts = glob.glob(all_txts)

	out_file = os.path.join(path_to_out, args.out)
	file = open(out_file, 'w+')
	print('image_name' + '\t' + 'x_coord' + '\t' + 'z_coord' + '\t' + 'y_coord' + '\t' + 'score', file=file)
	for f in txts:
		name = f.split('/')[-1][:-4]
		# print('name', name)
		
		with open(f) as dets:
			for i, l in enumerate(dets):
				if i > 0:
					l = l.split()
					# if l[]
					line = [name]
					line.extend(l)
					# print('line', line)
					file.write('\t'.join(line) + '\n')
	file.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for merge all coordinates')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)

