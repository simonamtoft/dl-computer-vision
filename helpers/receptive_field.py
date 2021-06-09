import numpy as np


def compute_receptive_field_(conv_dim, maxpool_idx, point):
    """
    Computes respective field indices of a point in the input image
    If conv_dim is from config, then conv_dim=config['conv_dim'][1:]
    """
    res_range = np.array([[point[0], point[0]], [point[1], point[1]]])
    for i in range(len(conv_dim)-1, -1, -1):
        kern = conv_dim[i][1]
        upd = np.array([[0, kern-1], [0, kern-1]])
        res_range += upd
        if i in maxpool_idx:
            print(res_range)
            res_range *= 2
            res_range += np.array([[0, 1], [0, 1]])
        print(res_range)
    return res_range


def compute_receptive_field(config, point):
    """Computes respective field indices of a point in the input image"""
    compute_receptive_field_(
        config['conv_dim'][1:], 
        config['maxpool_idx'], 
        point
    )