from __future__ import print_function,division

import numpy as np

def convert_3d_to_1d_coord(coords, w, h):
    num_coords = coords.shape[0]
    placeholder = np.zeros(num_coords)
    for i in range(num_coords):
        coord = coords[i]
        ind = coord[2] * (w * h) + coord[1] * w + coord[0]
        placeholder[i] = ind
    return placeholder

def coordinates_table_to_dict(coords):
    root = {}
    if 'source' in coords:
        for (source,name),df in coords.groupby(['source', 'image_name']):
            xy = df[['x_coord','y_coord', 'z_coord']].values.astype(np.int32)
            root.setdefault(source,{})[name] = xy
    else:
        for name,df in coords.groupby('image_name'):
            xy = df[['x_coord','y_coord', 'z_coord']].values.astype(np.int32)
            root[name] = xy
    return root
def coordinates_table_to_dict_class(coords):
    root = {}
    if 'source' in coords:
        for (source,name),df in coords.groupby(['source', 'image_name']):
            xyc = df[['x_coord','y_coord', 'z_coord', 'class']].values.astype(np.int32)
            root.setdefault(source,{})[name] = xy
    else:
        for name,df in coords.groupby('image_name'):
            xyc = df[['x_coord','y_coord', 'z_coord', 'class']].values.astype(np.int32)
            root[name] = xyc
    return root

def match_coordinates_to_images(coord, images):
    coords = coordinates_table_to_dict(coord)
    null_coords = np.zeros((0,3), dtype=np.int32)

    matched = {}

    for name in images.keys():
        im = images[name]
        # print('im',im.shape)
        depth, height, width = im.shape
        xy = coords.get(name, null_coords)
        matched[name]= {}
        matched[name]['tomo'] = im 
        matched[name]['coord'] = xy
        inds = convert_3d_to_1d_coord(xy, width, height)
        # print('xy', xy)
        matched[name]['inds'] = inds
        # print('inds', inds)
    return matched

def match_coordinates_class_to_images(coord, images):
    coords = coordinates_table_to_dict_class(coord)
    null_coords = np.zeros((0, 4), dtype = np.int32)
    matched = {}

    for name in images.keys():
        im = images[name]
        xyc = coords.get(name, null_coords)
        matched[name] = {}
        matched[name]['tomo'] = im 
        matched[name]['coord'] = xyc 

    return matched
