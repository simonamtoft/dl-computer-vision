import torch
import torch.nn.functional as F
from helpers import compute_metrics


def pad_output(y_pred, y_real):
    pad_size = (y_real.shape[2] - y_pred.shape[2])//2
    return F.pad(y_pred, (pad_size, pad_size, pad_size, pad_size))


def update_metrics(metrics, y_pred, y_real, n, threshold=0.5):
    # Get predictions
    y_pred = torch.sigmoid(y_pred).detach().cpu()
    y_pred = y_pred > threshold

    # Update metrics
    upd_metrics = compute_metrics(y_pred, y_real.cpu())
    metrics += upd_metrics / n
    return metrics
    

def remove_anno_dim(y_real):
    if y_real.ndim > 4:
        y_real = y_real[:, 0, :, :]
    return y_real
