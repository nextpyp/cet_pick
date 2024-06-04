import numpy as np 
import seaborn as sns
import pandas as pd  
import matplotlib
import json
import os 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import random_projection
from matplotlib import rcParams as rcp
import torchvision.transforms.functional as functional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch 
import matplotlib.offsetbox as osb
from cet_pick.colormap import ColorMap2DBremm,ColorMap2DZiegler, ColorMap2DSteiger,ColorMap2DSchumann
from matplotlib.patches import Circle
import matplotlib.patches as patches 
from sklearn.cluster import SpectralClustering,DBSCAN,KMeans
import argparse
import umap
import faiss 
# import faiss
import pickle
import collections



def quantize(x, mi=-3, ma=3, dtype=np.uint8):
    if mi is None:
        mi = x.min()
    if ma is None:
        ma = x.max()
    r = ma - mi
    x = 255*(x - mi)/r
    x = np.clip(x, 0, 255)
    x = np.round(x).astype(dtype)
    return x

def add_arguments(parser):
    parser.add_argument('--input', help='path to file containing coordinates, patches, embeddings')
    parser.add_argument('--n_cluster', type=int, help='number of clusters for overclustering')
    parser.add_argument('--num_neighbor', type=int, help='number of neighbors for both tsne and umap clustering')
    parser.add_argument('--mode', choices=['tsne', 'umap'], default='umap', help='choice of dimensionality reduction technique(default: umap)')
    parser.add_argument('--seed', type=int, default=42, help='file to output max stats')
    parser.add_argument('--host', type=int, default=7000, help='local host for images')
    parser.add_argument('--path', help='path of directory to save all output and images')
    parser.add_argument('--min_dist_umap', type=float, default=0.5, help='min distance for umap parameters')
    parser.add_argument('--save_out_img', type=int, default=1, help='whether to save images to output, if not saving img, use any number other than 1')
    parser.add_argument('--min_dist_vis', type=float,help='min distance for patch display on2d visualization')
    parser.add_argument('--gpus', default='0',   help='-1 for CPU, 0 for gpu')
    return parser

def main(args):
    data = np.load(args.input)
    patches = data['subvol']
    projs = data['pred']
    names = data['name']
    coords = data['coords']
    cmap = ColorMap2DZiegler()
    ncentroids = 256
    niter = 300
    verbose = True 
    d = projs.shape[1]
    is_gpu = False
    if args.gpus == '0':
        is_gpu = True 

    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=is_gpu, seed=1234)
    kmeans.train(projs)
    D, I = kmeans.index.search(projs, 1)
    centroids_kmeans = kmeans.centroids 
    spec_clustering = SpectralClustering(n_clusters=args.n_cluster,assign_labels='discretize', random_state=0)
    spec_clustering.fit(centroids_kmeans)
    y_pred = spec_clustering.labels_
    n_clusters_ = len(set(y_pred))
    print('Actual number of clusters is:', n_clusters_)
    final_lbs = []
    for temp_i in I:
        temp_i = temp_i[0]
        final_pred = y_pred[temp_i]
        final_lbs.append(final_pred)

    final_lbs = np.asarray(final_lbs)
    names_list = list(names)
    coords_list = [list(i) for i in coords]
    coords_list = [[str(j) for j in i] for i in coords_list]
    proj_list = [list(i) for i in projs]
    lb_list = list(final_lbs)
    out_img_path = os.path.join(args.path, 'imgs')
    if not os.path.exists(out_img_path):
        os.mkdir(out_img_path)
    rela_path = 'http://localhost:{}/imgs/'.format(args.host)
    all_paths = []
    for i in range(patches.shape[0]):
        curr_patch = patches[i]
        name = str(i)+'.png'
        out_path = os.path.join(out_img_path, name)
        reout_path = os.path.join(rela_path, name)
        all_paths.append(reout_path)
        if args.save_out_img == 1:
            plt.imsave(out_path, curr_patch[0], cmap='gray')

    all_info_dict = {"name": names_list, "coord": coords_list, "embeddings": proj_list, "label": lb_list,'image': all_paths}
    all_info_df = pd.DataFrame.from_dict(all_info_dict)
    out_parquet = os.path.join(args.path, 'interactive_info_parquet.gzip')
    all_info_df.to_parquet(out_parquet, compression='gzip')

    print('Done saving parquet')
    print('Plotting 2D visualization plot')
    if args.mode == 'tsne':
        projection= TSNE(n_components=2, perplexity = args.num_neighbor, verbose=1, random_state=args.seed, n_iter=1000)
    if args.mode == 'umap':
        projection = umap.UMAP(n_neighbors=args.num_neighbor, min_dist=args.min_dist_umap,random_state=args.seed)
    embeddings_2d = projection.fit_transform(projs)
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    colors = []
    embeddings_2d = (embeddings_2d - m) / (M - m)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(1, 1, 1)
    # shuffle images and find out which images to show
    shown_images_idx = []
    # shown_images = np.array([[1., 1.]])
    shown_images = np.expand_dims(embeddings_2d[0], axis=0)
    iterator = [i for i in range(embeddings_2d.shape[0])]
    for i in iterator: 
        # only show image if it is sufficiently far away from the others
        color = cmap(embeddings_2d[i][0], embeddings_2d[i][1])

        colors.append(color)
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < args.min_dist_vis:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)
    color_out = os.path.join(args.path, 'all_colors.npy')
    np.save(color_out, colors)
    print('Saved color outpot for 3D tomogram visualization')
    for idx in shown_images_idx:
#         # print('figure size', rcp['figure.figsize'][0])
#     # lb = final_lbs[idx]

        thumbnail_size = int(rcp['figure.figsize'][0] * 5)
        img = patches[idx]
        img = (img - img.mean())/img.std()
        img = quantize(img)
        img = torch.tensor(img)
        img = functional.resize(img, thumbnail_size)
        img = np.array(img)
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img[0], cmap=plt.cm.gray),
            embeddings_2d[idx],
            pad=0.2,
        )
        color = cmap(embeddings_2d[idx][0], embeddings_2d[idx][1])/255
        c = Circle(embeddings_2d[idx],0.03,fill=True,color=color)
        ax.add_artist(img_box)
        ax.add_patch(c)
    ratio = 1. / ax.get_data_ratio()

    ax.set_aspect('equal', adjustable='box')

# # figname = os.path.join(out_path, out_name) + '_landscape.png'
    out_2d_name = os.path.join(args.path, '2d_visualization_out.png')
    fig.savefig(out_2d_name)

    projection_lb = umap.UMAP(n_neighbors=args.num_neighbor, min_dist=args.min_dist_umap,random_state=args.seed)

    embeddings_2d = projection_lb.fit_transform(projs, y = final_lbs)
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(1, 1, 1)
    # shuffle images and find out which images to show
    shown_images_idx = []
    # shown_images = np.array([[1., 1.]])
    shown_images = np.expand_dims(embeddings_2d[0], axis=0)
    iterator = [i for i in range(embeddings_2d.shape[0])]
    for i in iterator: 
        # only show image if it is sufficiently far away from the others
        # color = cmap(embeddings_2d[i][0], embeddings_2d[i][1])

        # colors.append(color)
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < args.min_dist_vis:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)
    for idx in shown_images_idx:
#         # print('figure size', rcp['figure.figsize'][0])
        lb = final_lbs[idx]

        thumbnail_size = int(rcp['figure.figsize'][0] * 5)
        img = patches[idx]
        img = (img - img.mean())/img.std()
        img = quantize(img)
        img = torch.tensor(img)
        img = functional.resize(img, thumbnail_size)
        img = np.array(img)
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img[0], cmap=plt.cm.gray),
            embeddings_2d[idx],
            pad=0.2,
        )
        
        coord_txt = '%d' % (lb)
        ax.add_artist(img_box)
        ax.text(embeddings_2d[idx][0]-0.015, embeddings_2d[idx][1]+0.015, coord_txt)
    ratio = 1. / ax.get_data_ratio()

    ax.set_aspect('equal', adjustable='box')

    out_2d_name = os.path.join(args.path, '2d_visualization_labels.png')
    fig.savefig(out_2d_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for 2D visualization and necessary outputs for 3D visualization and interactive session.')
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
