import pandas as pd
import phoenix as px
import numpy as np 
import glob
import argparse
import os

def add_arguments(parser):
    parser.add_argument('--input',type=str, help='path to folder of all exported parquet from interactive session')
    parser.add_argument('--output', type=str, help='output with all training coordinates')
    parser.add_argument('--if_double', action='store_true', help='if need to double z coord (from compress to uncompress)')
    return parser 

def main(args):
	all_parquet = glob.glob(os.path.join(args.input, '*.parquet'))
	print('using the following parquet...', all_parquet)
	file = open(args.output,'w')
	header = ['image_name',	'x_coord','y_coord','z_coord']
	file.write('\t'.join(header)+'\n')
	for pq in all_parquet:
		df_p = pd.read_parquet(pq)
		names = df_p.loc[:,"name"].to_numpy()
		coords = df_p.loc[:,"coord"].to_numpy()
		for i in range(len(names)):
			n = names[i]
			x,y,z = coords[i]

			if args.if_double:
				z = str(float(z) * 2)
			line = [n,x,y,z]
			file.write('\t'.join(line)+'\n')
	file.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser('Script for converting interactive session exported parquet into training coordinates for refinement module')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)

