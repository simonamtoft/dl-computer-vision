import torch.nn as nn


class PureCNN(nn.Module):
    def __init__(self, config):
        super(PureCNN, self).__init__()
        
        in_channels = 3
        out_channels = 11

        conv_dims = [in_channels, *config['conv_dim'], out_channels]

        # Create list of conv layers
        n_pools = 0
        conv_layers = []
        for i in range(1, len(conv_dims)):
            # Add conv layer
            cd = conv_dims[i-1]
            conv_layers.append(
                nn.Conv2d(cd[0], cd[0], kernel_size=cd[1], padding=1)
            )

            # Add maxpool
            if i in config['maxpool_idx']:
                conv_layers.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                n_pools += 1
            
            # Add batchnorm
            if config['batch_norm']:
                conv_layers.append(nn.BatchNorm2d(conv_dims[i]))
            
            # Add ReLU activation
            conv_layers.append(nn.ReLU())

        # Define convolutional part
        self.conv = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        return self.conv(x)


class StandardCNN(nn.Module):
    def __init__(self, config):
        super(StandardCNN, self).__init__()
        
        conv_dim = config['conv_dim']
        fc_dim = config['fc_dim']

        # Create list of conv layers
        n_pools = 0
        conv_layers = [nn.Conv2d(3, conv_dim[0], kernel_size=3, padding=1), nn.ReLU()]
        for i in range(1, len(conv_dim)):
            if conv_dim[i-1] != conv_dim[i]:
                conv_layers.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                n_pools += 1

            conv_layers.append(
                nn.Conv2d(conv_dim[i-1], conv_dim[i], kernel_size=3, padding=1)
            )

            if config['batch_norm']:
                conv_layers.append(nn.BatchNorm2d(conv_dim[i]))

            conv_layers.append(nn.ReLU())

        
        # Create list of fully-connected layers
        fc_dims = [(128 // (2**n_pools))**2 * conv_dim[-1], *fc_dim, 2]
        fc_layers = []
        for i in range(len(fc_dims)-1):
            fc_layers.append(
                nn.Linear(fc_dims[i], fc_dims[i+1])
            )
            if i < len(fc_dims)-2:
                fc_layers.append(
                    nn.ReLU()
                )

        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x