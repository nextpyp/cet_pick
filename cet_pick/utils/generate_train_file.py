import glob 
import numpy as np 
import os 
import sys 
import argparse 

def add_arguments(parser):
    parser.add_argument('-e','--ext',help='micrograph extension', default=".rec")
    parser.add_argument('-d','--dir',help = 'path to training directory')
    parser.add_argument('-o', '--out',help='output name')
    parser.add_argument('-r', '--ord', help='order for x-y-z', default = 'xzy')   
    parser.add_argument('-c', '--inference', action='store_true', help='whether to generate files for inference')
    return parser 


def main(args):
    if args.inference:
        out_img_file = args.out + '_test_imgs.txt'
    else:
        out_img_file = args.out + '_train_imgs.txt'
    out_img_file = os.path.join(args.dir, out_img_file)
    if not args.inference:
        out_coordinates_file = args.out + '_train_coords.txt'
        out_coordinates_file = os.path.join(args.dir, out_coordinates_file)
    f1 = open(out_img_file, 'w')
    header = ['image_name','rec_path']
    len_ext = len(args.ext)
    f1.write('\t'.join(header) + '\n')
    all_mrcs = os.path.join(args.dir, "*"+args.ext)
    mrcs = glob.glob(all_mrcs)
    for path in mrcs:
        name = os.path.basename(path)[:-len_ext]
        # name = name[:-4]
        ent = [name, path]
        f1.write('\t'.join(ent) + '\n')
    f1.close()

    all_txts = os.path.join(args.dir, "*.txt")
    txt_f = glob.glob(all_txts)

    if not args.inference:
        f2 = open(out_coordinates_file, 'w')
        header = ['image_name','x_coord', 'y_coord', 'z_coord']
        f2.write('\t'.join(header) + '\n')
    
    
        for path in txt_f:
            name = os.path.basename(path)[:-4]
            if name[-10:] != 'train_imgs':
                print(name[-10:])
                with open(path, 'rb') as f:
                    for j, i in enumerate(f):
                        i = i.split()
                        if args.ord == 'xzy':
                            x = str(int(float(i[0])))

                            z = str(int(float(i[1])))
                            y = str(int(float(i[2])))
                        elif args.ord == 'xyz':
                            x = str(int(float(i[0])))

                            z = str(int(float(i[2])))
                            y = str(int(float(i[1])))
                        elif args.ord =='zxy':
                            x = str(int(float(i[1])))

                            z = str(int(float(i[0])))
                            
                            y = str(int(float(i[2])))
                        ent = [name, x, y, z]
                        f2.write('\t'.join(ent) + '\n')
        f2.close()


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for generating training and testing files')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
