import torch
import torch.nn as nn


def dice_loss(y_pred, y_real):
    y_pred = torch.sigmoid(y_pred)
    num = torch.mean(2 * y_real * y_pred + 1)
    den = torch.mean(y_real + y_pred + 1)
    return 1 - (num / den)


def bce_loss(y_pred, y_real):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))


def bce_total_variation(y_pred, y_real):
    y_hat = torch.sigmoid(y_pred)
    term1 = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
    term2 = y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]

    L_contiguity = torch.mean(torch.abs(term1)) + torch.mean(torch.abs(term2))

    return bce_loss(y_pred, y_real) + 0.1*L_contiguity


def focal_loss(y_pred, y_real, gamma=2):
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    term1 = (1 - y_pred)**gamma * y_real * torch.log(y_pred)
    term2 = (1 - y_real) * torch.log(1 - y_pred)
    return -torch.mean(term1 + term2)


def loss_func(loss='ce'):
    # # reshape if needed
    # if y_pred.ndim > 2:
    #     y_pred = torch.reshape(y_pred, (-1, y_pred.shape[1]))
    
    # get specified loss function
    if loss == 'ce':
        return nn.CrossEntropyLoss()
    elif loss == 'dice':
        return dice_loss
    elif loss == 'bce':
        return bce_loss
    elif loss == 'bce_var':
        return bce_total_variation
    elif loss == 'focal':
        return focal_loss
    else:
        raise Exception('Specified loss function not implemented!')
