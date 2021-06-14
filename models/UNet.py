import torch
import torch.nn as nn
from torch.nn.modules import module
from torchvision.transforms.functional import center_crop


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = 1
        n_labels = 1
        channels = config['channels']

        # encoder (downsampling)
        enc_dims = [in_channels, *channels]
        self.enc_conv = nn.ModuleList([])
        self.enc_pool = nn.ModuleList([])
        for i in range(1, len(enc_dims)):
            module_list = append_layer([], enc_dims[i-1], enc_dims[i], config)
            for _ in range(config['n_convs']-1):
                module_list = append_layer(module_list, enc_dims[i], enc_dims[i], config)

            self.enc_conv.append(nn.Sequential(*module_list))
            self.enc_pool.append(
                 nn.Conv2d(enc_dims[i], enc_dims[i], kernel_size=2, padding=0, stride=2)
            )

        # bottleneck
        module_list = []
        for _ in range(config['n_convs']-1):
            module_list.append(nn.Conv2d(enc_dims[i], enc_dims[i], kernel_size=3, padding=0))
            module_list.append(nn.ReLU())
        self.bottleneck = nn.Sequential(*module_list)

        # decoder (upsampling)
        dec_dims = [*channels]
        self.dec_conv = nn.ModuleList([])
        self.dec_upsample = nn.ModuleList([])
        for i in range(1, len(dec_dims)):
            module_list = append_layer([], dec_dims[i-1], dec_dims[i], config)
            for _ in range(config['n_convs']-1):
                module_list = append_layer(module_list, dec_dims[i], dec_dims[i], config)            
            self.dec_conv.append(nn.Sequential(*module_list))
            self.dec_upsample.append(
                nn.ConvTranspose2d(dec_dims[i], dec_dims[i], kernel_size=4, stride=2)
            )
        # final layer is without ReLU activation.
        self.dec_conv.append(nn.Sequential(
            nn.Conv2d(2*dec_dims[i], dec_dims[i], kernel_size=1, padding=0),
            nn.Conv2d(dec_dims[i], n_labels, kernel_size=1, padding=0),
            nn.Conv2d(n_labels, n_labels, kernel_size=1, padding=0)
        ))
        self.dec_upsample.append(
            nn.ConvTranspose2d(dec_dims[i], dec_dims[i], kernel_size=4, stride=2)
        )

    def forward(self, x):
        enc = x # x is input of first encoder
        
        # Pass through the encoder
        enc_out = []
        for i in range(len(self.enc_conv)):
            # Pass through a single encoder layer
            enc = self.enc_conv[i](enc)

            # save the encoder output such that it can be used for skip connections
            enc_out.append(enc)

            # Downsample with convolutional pooling
            enc = self.enc_pool[i](enc)

        # Pass through the bottleneck
        b = self.bottleneck(enc)

        # Pass through the decoder
        dec = b
        enc_out.reverse()   # reverse such that it fits pass through decoder
        for i in range(len(self.dec_conv)):
            # Get input for decoder
            dec = self.dec_upsample[i](dec)
            enc = enc_out[i]
            dec = skip_connection(enc, dec)

            # Pass through a single decoder layer
            dec = self.dec_conv[i](dec)
        
        return dec


def skip_connection(enc, dec):
    dec = center_crop(dec, output_size=enc.shape[2:3])
    return torch.cat([enc, dec], 1)


def append_layer(module_list, dim_1, dim_2, config):
    out_list = module_list

    # Add convolutional layer
    out_list.append(
        nn.Conv2d(dim_1, dim_2, kernel_size=3, padding=0)
    )

    # add batch norm
    if config['batch_norm']:
        out_list.append(
            nn.BatchNorm2d(dim_2)
        )

    # add activation
    out_list.append(nn.ReLU())

    # add dropout
    if config['dropout']:
        out_list.append(nn.Dropout(p=config['dropout']))
    
    return out_list
