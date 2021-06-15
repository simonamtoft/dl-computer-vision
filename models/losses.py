import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dice_loss(y_pred, y_real):
    y_pred = torch.sigmoid(y_pred)
    num = torch.mean(2 * y_real * y_pred + 1)
    den = torch.mean(y_real + y_pred + 1)
    return 1 - (num / den)


def bce_loss(y_pred, y_real, weights=[1, 1]):
    # https://discuss.pytorch.org/t/solved-class-weight-for-bceloss/3114
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    term1 = y_real * torch.log(y_pred)
    term2 = (1 - y_real) * torch.log(1 - y_pred)
    if weights != None:
        bce = term1 * weights[1] + term2 * weights[0]
    else:
        bce = term1 + term2
    return torch.neg(torch.mean(bce))


def bce_total_variation(y_pred, y_real):
    y_hat = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    term1 = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
    term2 = y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]

    L_contiguity = torch.mean(torch.abs(term1)) + torch.mean(torch.abs(term2))

    return bce_loss(y_pred, y_real) + 0.1*L_contiguity


def focal_loss(y_pred, y_real, gamma=2):
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    term1 = (1 - y_pred)**gamma * y_real * torch.log(y_pred)
    term2 = (1 - y_real) * torch.log(1 - y_pred)
    return -torch.mean(term1 + term2)


def loss_func(config):
    if len(config['loss_func']) > 1:
        loss = config['loss_func'][0]
        weight = config['loss_func'][1]
    else:
        loss = config['loss_func']

    if loss == 'ce':
        weight = torch.Tensor(weight).to(device)
        return nn.CrossEntropyLoss(weight=weight)
    elif loss == 'dice':
        return dice_loss
    elif loss == 'bce':
        return bce_loss
    elif loss == 'bce_var':
        return bce_total_variation
    elif loss == 'bce_weight':
        weight = torch.Tensor([weight]).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    elif loss == 'focal':
        return focal_loss
    else:
        raise Exception('Specified loss function not implemented!')
