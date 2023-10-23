import numpy as np 
import argparse

def convert_pt_to_txt(infile, image_name, outfile):
	file = open(outfile, 'w')
	header = ['image_name', 'x_coord', 'y_coord', 'z_coord', 'class']
	file.write('\t'.join(header) + '\n')
	with open(infile, 'r') as sp:
		for line in sp:
			line = line.split()
			clas = str(int(line[0]))
			x, z, y = line[2], line[3], line[4]
			ln = [image_name, str(x), str(y), str(z), clas]
			file.write('\t'.join(ln) + '\n')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'input coordinate files')
	parser.add_argument('-infile', help='input', type=str)
	parser.add_argument('-image', help='image_name', type=str)
	parser.add_argument('-outfile', help='outfile', type=str)

	args = parser.parse_args()

	convert_pt_to_txt(args.infile, args.image, args.outfile)
