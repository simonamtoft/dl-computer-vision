import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dice_loss(y_pred, y_real):
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    num = torch.mean(2 * y_real * y_pred + 1)
    den = torch.mean(y_real + y_pred + 1)
    return 1 - (num / den)


def bce_loss(y_pred, y_real, weight):
    # ensure stability
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)

    # compute loss terms
    term1 = y_real * torch.log(y_pred)
    term2 = (1 - y_real) * torch.log(1 - y_pred)

    # weight the terms
    term1 *= weight[1]
    term2 *= weight[0]

    # return loss as negative mean of sum of the terms
    return torch.neg(torch.mean(term1 + term2))


class BCELoss(nn.Module):
    def __init__(self, weight):
        self.weight = weight
    
    def forward(self, y_pred, y_real):
        return bce_loss(y_pred, y_real, self.weight)


class BCEVarLoss(nn.Module):
    def __init__(self, weight):
        self.weight = weight
        self.cont_weight = 0.1
    
    def forward(self, y_pred, y_real):
        # Compute continuity
        y_hat = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
        term1 = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
        term2 = y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]
        L_contiguity = torch.mean(torch.abs(term1)) + torch.mean(torch.abs(term2))
        
        # Compute weighted BCE
        bce = bce_loss(y_pred, y_real, self.weight)

        # Add together
        return bce + self.cont_weight * L_contiguity


class FocalLoss(nn.Module):
    def __init__(self, weight):
        self.weight = weight
        self.gamma = 2
    
    def forward(self, y_pred, y_real):
        # ensure stability
        y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)

        # compute loss terms for each class
        term1 = (1 - y_pred)**self.gamma * y_real * torch.log(y_pred)
        term2 = (1 - y_real) * torch.log(1 - y_pred)
        
        # add weights to terms
        term1 *= self.weight[1]
        term2 *= self.weight[0]
        
        # return loss as negative mean of sum
        return torch.neg(torch.mean(term1 + term2))


def loss_func(config):
    if len(config['loss_func']) > 1:
        loss = config['loss_func'][0]
        weight = torch.Tensor(config['loss_func'][1]).to(device)
    else:
        loss = config['loss_func']

    if loss == 'ce':
        return nn.CrossEntropyLoss(weight=weight)
    elif loss == 'dice':
        return dice_loss
    elif loss == 'bce':
        return BCELoss(weight=weight)
    elif loss == 'bce_var':
        return BCEVarLoss(weight=weight)
    elif loss == 'bce_weight':
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    elif loss == 'focal':
        return FocalLoss(weight=weight)
    else:
        raise Exception('Specified loss function not implemented!')
