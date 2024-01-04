from __future__ import print_function, division

import os

import numpy as np
from PIL import Image

import torch
import torch.utils.data


def enumerate_pn_coordinates_hm(Y, thresh):
   
    P_size = sum((y>thresh).sum() for y in Y) # number of positive coords (gaussian dist)
    N_size = sum(y.size for y in Y) - P_size  

    P = np.zeros(P_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    N = np.zeros(N_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    i = 0 # P index
    j = 0 # N index

    for image in range(len(Y)):
        y = Y[image].ravel()
        for coord in range(len(y)):
            if y[coord] > thresh:
                P[i] = (image, coord)
                i += 1
            else:
                N[j] = (image, coord)
                j += 1

    return P, N

def enumerate_pu_coordinates_hm(Y, thresh):
    P_size = sum((y>thresh).sum() for y in Y) # number of positive coords (gaussian dist)
    size = sum(y.size for y in Y)

    P = np.zeros(P_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    U = np.zeros(size, dtype=[('image', np.uint32), ('coord', np.uint32)])

    i = 0 # P index
    j = 0 # U index
    for image in range(len(Y)):
        y = Y[image].ravel()
        for coord in range(len(y)):
            if y[coord] > thresh:
                P[i] = (image, coord)
                i += 1
            U[j] = (image, coord)
            j += 1

    return P, U

def enumerate_pn_coordinates(Y, tomos):
    """
    Given a list of 2d arrays containing labels, enumerate the positive and negative coordinates as (image,coordinate) pairs.
    """

    # P_size = int(sum(y.sum() for y in Y)) # number of positive coordinates
    P_size = sum(len(y) for y in Y) # number of positive coords
    A_size = sum(y.size for y in tomos) #number of all coords
    N_size = A_size - P_size # number of negative coordinates
    P = np.zeros(P_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    N = np.zeros(N_size, dtype=[('image', np.uint32), ('coord', np.uint32)])

    i = 0 # P index
    j = 0 # N index
    # img_size = tomos[]
    # all_coords = np.arange(0, A_size)
    for image in range(len(Y)):
        # y = Y[image].ravel()
        tomo = tomos[image]
        tot_coords = tomo.size
        all_coords = np.zeros(tot_coords)
        # all_coords = set(np.arange(0, tot_coords))
        y = Y[image]
        set_y = set(y)
        # for y in Y:
        for coord in all_coords:
            if coord in set_y:
                P[i] = (image, coord)
                i += 1
            else:
                N[j] = (image, coord)
                j += 1
    # re

        # for coord in all_coords:

        # for coord in range(len(y)):
        #     if y[coord]:
        #         P[i] = (image, coord)
        #         i += 1
        #     else:
        #         N[j] = (image, coord)
        #         j += 1

    return P, N

def enumerate_pu_coordinates(Y, tomos):
    """
    Given a list of 3d arrays containing labels, enumerate the positive and unlabeled(all) coordinates as (image,coordinate) pairs.
    """

    # P_size = int(sum(y.sum() for y in Y)) # number of positive coordinates
    # size = sum(y.size for y in Y)
    P_size = sum(len(y) for y in Y) # number of positive coords
    A_size = sum(y.size for y in tomos) #number of all coords
    P = np.zeros(P_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    U = np.zeros(A_size, dtype=[('image', np.uint32), ('coord', np.uint32)])

    i = 0 # P index
    j = 0 # U index
    for image in range(len(Y)):
        # y = Y[image].ravel()
        tomo = tomos[image]
        tot_coords = tomo.size
        all_coords = np.zeros(tot_coords)
        # all_coords = set(np.arange(0, tot_coords))
        y = Y[image]
        set_y = set(y)
        for coord in all_coords:
            if coord in set_y:
                P[i] = (image, coord)
                i += 1
            U[j] = (image, coord)
            j += 1
        # for coord in range(len(y)):
        #     if y[coord]:
        #         P[i] = (image, coord)
        #         i += 1
        #     U[j] = (image, coord)
        #     j += 1

    return P, U

class ShuffledSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, x, random=np.random):
        self.x = x
        self.random = random
        self.i = len(self.x)

    def __len__(self):
        return len(self.x)

    def __next__(self):
        if self.i >= len(self.x):
            self.random.shuffle(self.x)
            self.i = 0
        sample = self.x[self.i]
        self.i += 1
        return sample

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        return self

class StratifiedCoordinateHMSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels, thresh=0.5, balance=0.5, size=None, random=np.random, split='pn'):
        groups = []
        weights = np.zeros(2)
        proportions = np.zeros((1, 2))

        i = 0
        if split == 'pn':
            P,N = enumerate_pn_coordinates_hm(labels, thresh)
            P = ShuffledSampler(P, random=random)
            N = ShuffledSampler(N, random=random)
            groups.append(P)
            groups.append(N)

            proportions[i//2,0] = len(N)/(len(N)+len(P))
            proportions[i//2,1] = len(P)/(len(N)+len(P))
        elif split == 'pu':
            P, U = enumerate_pu_coordinates_hm(labels, thresh)
            P = ShuffledSampler(P, random=random)
            U = ShuffledSampler(U, random=random)
            groups.append(P)
            groups.append(U)

            proportions[i//2,0] = (len(U) - len(P))/len(U)
            proportions[i//2,1] = len(P)/len(U)

        p = balance
        if balance is None:
            p = proportions[i//2,1]
        # weights[i] = p/len(labels)
        weights[i] = p
        # weights[i+1] = (1-p)/len(labels)
        weights[i+1] = 1-p

        if size is None:
            # sizes = np.array([len(g) for g in groups])
            sizes = np.array(len(tomos))
            size = int(np.round(np.min(sizes/weights)))

        self.groups = groups
        self.weights = weights
        self.proportions = proportions
        self.size = size

        self.history = np.zeros_like(self.weights)
        self.random = random
        
    def __len__(self):
        return self.size

    def __next__(self):
        n = self.history.sum()
        weights = self.weights
        if n > 0:
            weights = weights - self.history/n
            weights[weights < 0] = 0
            n = weights.sum()
            if n > 0:
                weights /= n
            else:
                weights = np.ones_like(weights)/len(weights)

        i = self.random.choice(len(weights), p=weights)
        self.history[i] += 1
        if np.all(self.history/self.history.sum() == self.weights):
            self.history[:] = 0

        g = self.groups[i]
        sample = next(g)

        i = i//2
        j,c = sample

        # code as integer
        # unfortunate hack required because pytorch converts index to integer...

        h = i*2**56 + j*2**32 + c
        # print('h in sampler', h)
        return h
        #return i//2, sample

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        for _ in range(self.size):
            yield next(self)

class StratifiedCoordinateSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels, tomos, balance=0.5, size=None, random=np.random, split='pn'):

        groups = []
        weights = np.zeros(2)
        proportions = np.zeros((1, 2))
        # i = 0

        # for group in labels:
        if split == 'pn':
            P,N = enumerate_pn_coordinates(labels, tomos)
            P = ShuffledSampler(P, random=random)
            N = ShuffledSampler(N, random=random)
            groups.append(P)
            groups.append(N)

            proportions[i//2,0] = len(N)/(len(N)+len(P))
            proportions[i//2,1] = len(P)/(len(N)+len(P))
        elif split  == 'pu':
            P,U = enumerate_pu_coordinates(labels, tomos)
            P = ShuffledSampler(P, random=random)
            U = ShuffledSampler(U, random=random)
            groups.append(P)
            groups.append(U)

            proportions[i//2,0] = (len(U) - len(P))/len(U)
            proportions[i//2,1] = len(P)/len(U)

        p = balance
        if balance is None:
            p = proportions[i//2,1]
        # weights[i] = p/len(labels)
        weights[i] = p
        # weights[i+1] = (1-p)/len(labels)
        weights[i+1] = 1-p
        # i += 2

        if size is None:
            # sizes = np.array([len(g) for g in groups])
            sizes = np.array(len(tomos))
            size = int(np.round(np.min(sizes/weights)))

        self.groups = groups
        self.weights = weights
        self.proportions = proportions
        self.size = size

        self.history = np.zeros_like(self.weights)
        self.random = random

    def __len__(self):
        return self.size

    def __next__(self):
        n = self.history.sum()
        weights = self.weights
        if n > 0:
            weights = weights - self.history/n
            weights[weights < 0] = 0
            n = weights.sum()
            if n > 0:
                weights /= n
            else:
                weights = np.ones_like(weights)/len(weights)

        i = self.random.choice(len(weights), p=weights)
        self.history[i] += 1
        if np.all(self.history/self.history.sum() == self.weights):
            self.history[:] = 0

        g = self.groups[i]
        sample = next(g)

        i = i//2
        j,c = sample

        # code as integer
        # unfortunate hack required because pytorch converts index to integer...

        h = i*2**56 + j*2**32 + c
        # print('h in sampler', h)
        return h
        #return i//2, sample

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        for _ in range(self.size):
            yield next(self)


