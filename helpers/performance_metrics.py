# Compute different performance metrics.
# Formulas found here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4614595/
import torch


def compute_dice(pred, anno):
    BATCH_SIZE = pred.shape[0]
    
    pred2 = torch.sigmoid(pred)
    num = torch.mean((2 * anno * pred2 + 1).view(BATCH_SIZE,-1),dim=1)
    den = torch.mean((anno + pred2 + 1).view(BATCH_SIZE,-1),dim=1)
    DICE =  1 - (num / den)
    #pred = torch.sigmoid(pred)
    #num = torch.mean(2 * anno * pred + 1)
    #den = torch.mean(anno + pred + 1)
    return DICE


def compute_iou(pred, anno):
    """Computes the intersection over union of the predicted and annotated segmentations"""
    BATCH_SIZE = pred.shape[0]
    
    A = pred == 1
    B = anno == 1
    C = torch.sum((A & B).view(BATCH_SIZE,-1),dim=1)
    D = torch.sum((A | B).view(BATCH_SIZE,-1),dim=1)
    nan_idx = D==1

    IoU = C/D
    IoU[nan_idx] = 0
    
    #A = pred == 1
    #B = anno == 1
    #return torch.sum(A & B) / torch.sum(A | B)
    return IoU


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
    nan_idx = (TP + FN) == 0
    sens = TP / (TP + FN)
    sens[nan_idx] = 0
    #if (TP + FN) == 0:
    #    return 0
    #return TP / (TP + FN)
    return sens


def compute_specificity(TN, FP):
    """Compute the specificity: TN / (TN + FP)
    Input
        TN  (int)   :   Number of true negatives predicted
        FP  (int)   :   Number of false positives predicted
        
    """
    nan_idx = (TN + FP) == 0
    spec = TN / (TN + FP)
    spec[nan_idx] = 0
    return spec
    #if (TN + FP) == 0:
    #    return 0
    #return TN / (TN + FP)


def compute_metrics(pred, anno):
    # Compute confusion metrics
    BATCH_SIZE = pred.shape[0]
    TP = torch.sum(((pred == 1) & (anno == 1)).view(BATCH_SIZE,-1),dim=1)
    TN = torch.sum(((pred == 0) & (anno == 0)).view(BATCH_SIZE,-1),dim=1)
    FP = torch.sum(((pred == 1) & (anno == 0)).view(BATCH_SIZE,-1),dim=1)
    FN = torch.sum(((pred == 0) & (anno == 1)).view(BATCH_SIZE,-1),dim=1)
    #TP = torch.sum((pred == 1) & (anno == 1))
    #TN = torch.sum((pred == 0) & (anno == 0))
    #FP = torch.sum((pred == 1) & (anno == 0))
    #FN = torch.sum((pred == 0) & (anno == 1))
    
    # Compute performance metrics
    dice = compute_dice(pred, anno)
    iou = compute_iou(pred, anno)
    acc = compute_accuracy(TP, TN, FP, FN)
    sens = compute_sensitivity(TP, FN)
    spec = compute_specificity(TN, FP)
    
    acc = torch.mean(acc)
    sens = torch.mean(sens)
    spec = torch.mean(spec)
    dice = torch.mean(dice)
    iou = torch.mean(iou)
    
    return (dice, iou, acc, sens, spec)
