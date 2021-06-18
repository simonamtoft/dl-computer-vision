import torch
import torch.nn as nn
import torch.nn.functional as F


class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self)

    def discriminator(self, d, x_real, x_fake):
        return None
    
    def generator(self, d, x_real, x_fake):
        return None


class LSGANLoss(nn.Module):
    def __init__(self, params):
        """Input list params of [a, b, c]"""
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
    
    def discriminator(self, d, x_real, x_fake):
        y_real = d(x_real)
        y_fake = d(x_fake.detach())
        term1 = torch.mean((y_real - self.b)**2)
        term2 = torch.mean((y_fake - self.a)**2)
        return 0.5 * (term1 + term2)
    
    def generator(self, d, x_real, x_fake):
        y_fake = d(x_fake)
        return 0.5 * torch.mean((y_fake - self.c)**2)


class MinimaxLoss(nn.Module):
    def __init__(self):
        super(MinimaxLoss, self)

    def discriminator(self, d, x_real, x_fake):
        E_real = F.logsigmoid(d(x_real))
        E_fake = F.logsigmoid(-d(x_fake.detach()))
        d_loss = -torch.mean(E_real + E_fake)
        return d_loss

    def generator(self, d, x_real, x_fake):
        return -torch.mean(F.logsigmoid(d(x_fake)))


def gan_loss_func(config):
    loss_name = config['loss_func'][0]

    if loss_name == 'minimax':
        return MinimaxLoss()
    elif loss_name == 'lsgan':
        return LSGANLoss(config['loss_func'][1])
    else:
        raise Exception(f"Specified loss function '{loss_name}' not implemented.")


def gan_im_loss(config):
    if config['img_loss'] == 'l1':
        return nn.L1Loss()
    elif config['img_loss'] == 'l2':
        return nn.MSELoss()
    else:
        raise Exception(f"Provided image loss {config['img_loss']} not defined.")
