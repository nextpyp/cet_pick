import pandas as pd
import numpy as np 
import os

def list_of_ints(arg):
  return list(map(int, arg.split(',')))

def add_arguments(parser):
    parser.add_argument('--input',type=str, help='path to interactive info parquet file')
    parser.add_argument('--out_path', type=str, help='output with all training coordinates')
    parser.add_argument('--if_double', action='store_true', help='if need to double z coord (from compress to uncompress)')
    parser.add_argument('--use_classes', type=list_of_ints, help='selected labels to use for trainning coordinates')
    return parser 


def main(args):
	num_labels = len(args.use_classes)
	print('selecting {} number of classes'.format(num_labels))
	info_df = pd.read_parquet(args.input)
	print(info_df.head())
	sub_df = info_df.loc[info_df['label'].isin(args.use_classes)]
	names = np.unique(sub_df['name'].to_numpy())
	for nm in names:
		out_full = os.path.join(args.out_path, nm+'.txt')
		file = open(out_full, 'w')
		subsubdf = sub_df.loc[sub_df['name'] == nm]
		for c in subsubdf.coord:
			x,y,z = float(c[0]), float(c[1]), float(c[2])
			if args.if_double:
				z *= 2
			line = [str(x), str(z), str(y)]
			file.write('\t'.join(line)+'\n')
		file.close()
	

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for extracting selected labels from generated parquet info file from plot_2d.py ')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)


