from __future__ import absolute_import, print_function, division

import numpy as np
from scipy.optimize import linear_sum_assignment

def match_coordinates(targets, preds, radius):
    d2 = np.sum((preds[:,np.newaxis] - targets[np.newaxis])**2, 2)
    cost = d2 - radius*radius*radius
    cost[cost > 0] = 0
    # print('cost', cost)
    pred_index,target_index = linear_sum_assignment(cost)

    cost = cost[pred_index, target_index]
    dist = np.zeros(len(preds))
    dist[pred_index] = np.sqrt(d2[pred_index, target_index])

    pred_index = pred_index[cost < 0]
    assignment = np.zeros(len(preds), dtype=np.float32)
    assignment[pred_index] = 1

    return assignment, dist