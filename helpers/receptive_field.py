import numpy as np


def compute_receptive_field_(conv_dim, maxpool_idx, point):
    """
    Computes respective field indices of a point in the input image
    Input:
        point:          Input point with coordinates [x, y]
        conv_dim:       List of lists of channels and kernel dimensions of model [[C, K]]
        maxpool_idx:    List of idx of maxpool layers in model.
    Returns:
        out : [height, left, top, width]
        res_range: [[x1, x2], [y1, y2]]
    """
    res_range = np.array([[point[0], point[0]], [point[1], point[1]]])
    for i in range(len(conv_dim)-1, -1, -1):
        kern = conv_dim[i][1]
        upd = np.array([[0, kern-1], [0, kern-1]])
        res_range += upd
        if i in maxpool_idx:
            res_range *= 2
            res_range += np.array([[0, 1], [0, 1]])
    width = res_range[0, 1] - res_range[0, 0]
    height = res_range[1, 1] - res_range[1, 0]
    out = np.array([height, res_range[0, 0], res_range[1, 0], width])
    return out, res_range


def compute_receptive_field(config, point):
    """
    Computes respective field indices of a point in the input image
    Input:
        point:          Input point with coordinates [x, y]
        config:         Config dict of model.
    Returns:
        out:        [height, left, top, width]
        res_range:  Range of receptive field [[x1, x2], [y1, y2]]
    """
    return compute_receptive_field_(
        config['conv_dim'], 
        config['maxpool_idx'], 
        point
    )


if __name__=="__main__":
    # example usage
    point = [0, 0]
    conv_dim = [[-1, 3], [-1, 3], [-1, 1], [-1, 1]] 
    max_idx = [1]
    config = {
        'conv_dim': conv_dim,
        'maxpool_idx': max_idx
    }
    print(compute_receptive_field(config, point))