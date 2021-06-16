# Compute different performance metrics.
# Formulas found here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4614595/
import torch
import numpy as np


def compute_dice(TP, FP, FN): 
    # Compute dice
    dice = 2 * TP / (2 * TP + FP + FN)

    # Stability
    nan_idx = (2 * TP + FP + FN) == 0
    dice[nan_idx] = 0

    return torch.mean(dice)


def compute_iou(pred, anno, batch_size):
    """Computes the intersection over union of the predicted and annotated segmentations"""
    # Get two areas
    A = pred == 1
    B = anno == 1
    
    # Compute IoU
    C = torch.sum((A & B).view(batch_size, -1), dim=1)
    D = torch.sum((A | B).view(batch_size, -1), dim=1)
    IoU = C/D
    
    # Ensure stability
    nan_idx = D == 0
    IoU[nan_idx] = 0

    return torch.mean(IoU)


def compute_accuracy(TP, TN, FP, FN):
    """Compute the accuracy: (TP + TN) / (TP + TN + FP + FN)
    Input
        TP  (int)   :   Number of true positives predicted
        TN  (int)   :   Number of true negatives predicted
        FP  (int)   :   Number of false positives predicted
        FN  (int)   :   Number of false negatives predicted
    """
    acc = (TP + TN) / (TP + TN + FP + FN)
    return torch.mean(acc)


def compute_sensitivity(TP, FN):
    """Compute the sensitivity: TP / (TP + FN)
    Inputs:
        TP  (int)   :   Number of true positives predicted
        FN  (int)   :   Number of false negatives predicted
    """
    # Compute sensitivity 
    sens = TP / (TP + FN)

    # Ensure stability
    nan_idx = (TP + FN) == 0
    sens[nan_idx] = 0 

    return torch.mean(sens)


def compute_specificity(TN, FP):
    """Compute the specificity: TN / (TN + FP)
    Input
        TN  (int)   :   Number of true negatives predicted
        FP  (int)   :   Number of false positives predicted
        
    """
    #  Compute specificity
    spec = TN / (TN + FP)

    # Ensure stability
    nan_idx = (TN + FP) == 0
    spec[nan_idx] = 0

    return torch.mean(spec)


def compute_metrics(pred, anno):
    batch_size = pred.shape[0]

    # Compute confusion metrics
    TP = torch.sum(((pred == 1) & (anno == 1)).view(batch_size, -1), dim=1)
    TN = torch.sum(((pred == 0) & (anno == 0)).view(batch_size, -1), dim=1)
    FP = torch.sum(((pred == 1) & (anno == 0)).view(batch_size, -1), dim=1)
    FN = torch.sum(((pred == 0) & (anno == 1)).view(batch_size, -1), dim=1)
    
    # Compute performance metrics
    dice = compute_dice(TP, FP, FN).numpy()
    iou = compute_iou(pred, anno, batch_size).numpy()
    acc = compute_accuracy(TP, TN, FP, FN).numpy()
    sens = compute_sensitivity(TP, FN).numpy()
    spec = compute_specificity(TN, FP).numpy()
    return np.array([dice, iou, acc, sens, spec])
