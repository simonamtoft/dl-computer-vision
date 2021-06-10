import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def non_maximum_supression(config, pred, output, p_thresh=0.8, iou_thresh=0.1):
    """ Returns list of boxes after performing non-maximum supression.
    Inputs:
        config      :   Config used to train model.
        pred        :   Argmax of model output with shape [H x W]
        output      :   Output of model (after softmax) with shape [C x H x W]
        p_thresh    :   Probability cut-off value [0; 1]. Higher is stricter.
        iou_thresh  :   IoU cut-off value [0; 1]. Lower is stricter.
    Returns:
    List of dicts with fields
        bbox        : [height, left, top, width]
        c           : Predicted class of bbox
        p           : Probability of prediction
        range       : [x1, x2, y1, y2]
    """
    raw_boxes = create_box_dict(config, pred, output, p_thresh)
    return prune_boxes_iou(raw_boxes, iou_thresh)


def create_box_dict(config, pred, output, p_thresh):
    boxes = [[] for i in range(10)]
    for y in range(pred.shape[0]):
        for x in range(pred.shape[1]):
            c = pred[y, x]
            if c != 10:
                p = output[c, y, x]
                if p > p_thresh:
                    bbox, ran = compute_receptive_field(config, [x, y])
                    boxes[c].append({
                        'p': p,
                        'c': c,
                        'range': ran.flatten(),
                        'bbox': bbox
                    })
    return boxes


def prune_boxes_iou(raw_boxes, iou_thresh):
    boxes = []
    for i, c_boxes in enumerate(raw_boxes):
        while c_boxes:
            max_box = max(c_boxes, key=lambda x: x["p"])
            boxes.append(max_box)
            c_boxes.remove(max_box)

            _c_boxes = []
            for box in c_boxes:
                if get_iou(max_box['range'], box['range']) <= iou_thresh:
                    _c_boxes.append(box)
            c_boxes = _c_boxes
    return boxes


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


def get_iou(bb1, bb2):
    """
    bb1: range of x and y such as [x1, x2, y1, y2]
    bb2: range of x and y such as [x1, x2, y1, y2]
    """
    assert bb1[0] < bb1[1]
    assert bb1[2] < bb1[3]
    assert bb2[0] < bb2[1]
    assert bb2[2] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    x_right = min(bb1[1], bb2[1])
    y_top = max(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    
    # Check if the two boxes have 0 overlap
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def plt_bboxes(img, boxes, title=""):
    """Plot bounding boxes on top of image
    Inputs
        img     :   Image of shape [3 x H x W]
        boxes   :   Bounding boxes from the function non_maximum_supression
    """
    f, ax = plt.subplots()
    ax.imshow(np.swapaxes(np.swapaxes(img.numpy(), 0, 2), 0, 1))
    colors = ['red', 'yellow', 'green', 'blue', 'orange', 'teal', 'pink', 'cyan', 'black', 'magenta']
    labels = []
    for box in boxes:
        rec = box['bbox']
        c = box['c'].numpy()

        if c not in labels:
            rect = patches.Rectangle(
                (rec[1], rec[2]), rec[3], rec[0], 
                linewidth=1, edgecolor=colors[box['c']], facecolor='none',
                label=f"{box['c']}"
            )
            labels.append(c)
        else:
            rect = patches.Rectangle(
                (rec[1], rec[2]), rec[3], rec[0], 
                linewidth=1, edgecolor=colors[box['c']], facecolor='none',
            )
        ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')
    ax.legend(loc='upper left')
    plt.plot()


if __name__=="__main__":
    # example usage of receptive field
    point = [0, 0]
    conv_dim = [[-1, 3], [-1, 3], [-1, 1], [-1, 1]] 
    max_idx = [1]
    config = {
        'conv_dim': conv_dim,
        'maxpool_idx': max_idx
    }
    print(compute_receptive_field(config, point))
