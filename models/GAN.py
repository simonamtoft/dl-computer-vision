import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

norm_layer = nn.InstanceNorm2d


class ResBlock(nn.Module):
    def __init__(self, f, sn):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            sn(nn.Conv2d(f, f, 3, 1, 1)), 
            norm_layer(f), nn.ReLU(),
            sn(nn.Conv2d(f, f, 3, 1, 1))
        )
        self.norm = norm_layer(f)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)+x))


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        f = config['n_features']
        blocks = config['n_blocks']

        if "spectral_norm" in config and config["spectral_norm"]:
            sn = spectral_norm
        else:
            sn = lambda x: x

        # Encoding layer
        layers = [
            nn.ReflectionPad2d(3),
            sn(nn.Conv2d(  3,   f, kernel_size=7, stride=1, padding=0)), 
            norm_layer(  f), nn.ReLU(True),
            sn(nn.Conv2d(  f, 2*f, kernel_size=3, stride=2, padding=1)), 
            norm_layer(2*f), nn.ReLU(True),
            sn(nn.Conv2d(2*f, 4*f, kernel_size=3, stride=2, padding=1)), 
            norm_layer(4*f), nn.ReLU(True)
        ]

        # Transformation (Resnet) layer
        for _ in range(int(blocks)):
            layers.append(ResBlock(4*f, sn))
        
        # Decoding layer
        # Uses a subpixel convolution (PixelShuffle) for upsamling 
        layers.extend([ 
                sn(nn.ConvTranspose2d(4*f, 4*2*f, kernel_size=3, stride=1, padding=1)), 
                nn.PixelShuffle(2), norm_layer(2*f), nn.ReLU(True),
                sn(nn.ConvTranspose2d(2*f,   4*f, kernel_size=3, stride=1, padding=1)), 
                nn.PixelShuffle(2), norm_layer(  f), nn.ReLU(True),
                nn.ReflectionPad2d(3), sn(nn.Conv2d(f, 3, 7, 1, 0)),
                nn.Tanh()
        ])
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        f = config['n_features']
        relu_val = config['relu_val']

        if "spectral_norm" in config and config["spectral_norm"]:
            sn = spectral_norm
        else:
            sn = lambda x: x

        self.conv = nn.Sequential(
            sn(nn.Conv2d(  3,   f, kernel_size=4, stride=2, padding=1)), 
            norm_layer(  f), 
            nn.LeakyReLU(relu_val, True),
            sn(nn.Conv2d(  f, 2*f, kernel_size=4, stride=2, padding=1)), 
            norm_layer(2*f), 
            nn.LeakyReLU(relu_val, True),
            sn(nn.Conv2d(2*f, 4*f, kernel_size=4, stride=2, padding=1)), 
            norm_layer(4*f),
            nn.LeakyReLU(relu_val, True),
            sn(nn.Conv2d(4*f, 8*f, kernel_size=4, stride=1, padding=1)), 
            norm_layer(8*f),
            nn.LeakyReLU(relu_val, True),
        )

        self.final = nn.Sequential(
            sn(nn.Conv2d(8*f, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        return self.final(self.conv(x))