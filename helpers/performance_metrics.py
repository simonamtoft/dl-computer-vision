# Compute different performance metrics.
# Formulas found here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4614595/
import torch


def compute_dice(pred, anno):
    pred = torch.sigmoid(pred)
    num = torch.mean(2 * anno * pred + 1)
    den = torch.mean(anno + pred + 1)
    return 1 - (num / den)


def compute_iou(pred, anno):
    """Computes the intersection over union of the predicted and annotated segmentations"""
    A = pred == 1
    B = anno == 1
    return torch.sum(A & B) / torch.sum(A | B)


def compute_accuracy(TP, TN, FP, FN):
    """Compute the accuracy: (TP + TN) / (TP + TN + FP + FN)
    Input
        TP  (int)   :   Number of true positives predicted
        TN  (int)   :   Number of true negatives predicted
        FP  (int)   :   Number of false positives predicted
        FN  (int)   :   Number of false negatives predicted
    """
    return (TP + TN) / (TP + TN + FP + FN)


def compute_sensitivity(TP, FN):
    """Compute the sensitivity: TP / (TP + FN)
    Inputs:
        TP  (int)   :   Number of true positives predicted
        FN  (int)   :   Number of false negatives predicted
    """
    return TP / (TP + FN)


def compute_specificity(TN, FP):
    """Compute the specificity: TN / (TN + FP)
    Input
        TN  (int)   :   Number of true negatives predicted
        FP  (int)   :   Number of false positives predicted
    """
    return TN / (TN + FP)


def compute_metrics(pred, anno):
    # Compute confusion metrics
    TP = torch.sum((pred == 1) & (anno == 1))
    TN = torch.sum((pred == 0) & (anno == 0))
    FP = torch.sum((pred == 1) & (anno == 0))
    FN = torch.sum((pred == 0) & (anno == 1))
    
    # Compute performance metrics
    dice = compute_dice()
    iou = compute_iou()
    acc = compute_accuracy(TP, TN, FP, FN)
    sens = compute_sensitivity(TP, FN)
    spec = compute_specificity(TN, FP)
    return (dice, iou, acc, sens, spec)
