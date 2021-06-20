import os
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from models import Generator, Discriminator
from training import train_cycle_gan
from data import HORSES, ZEBRAS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup training config
config = {
    'batch_size': 16,
    'epochs': 100,
    'lr_d': 4*1e-4,
    'lr_g': 2*1e-4,
    'loss_func': ['lsgan',[0,1,1]], # 'lsgan' or 'minimax'
    'g_loss_weight': [1, 10, 5],
    'n_features': 64,
    'n_blocks': 9,
    'relu_val': 0.2,
    'img_loss': ['l1', 'l1'], # cycle, identity 
    'buf_size': 0,
    'lr_decay': {
        'offset': 0,
        'delay': 20,
        'n_epochs': 100
    },
    'affine': True,
    'resize': True,
}

# Instantiate models
g_h2z = Generator(config).to(device)
g_z2h = Generator(config).to(device)
d_h = Discriminator(config).to(device)
d_z = Discriminator(config).to(device)

# Create transforms
transforms_list = [transforms.ToTensor()]
if config['resize']:
    transforms_list.append(
        transforms.Resize((128, 128))
    )
test_transform = transforms.Compose(transforms_list)
transforms_list = [transforms.ToTensor()]
if config['affine']:
    transforms_list.extend([
        transforms.Pad(10, padding_mode='reflect'),
        transforms.RandomAffine(degrees=7, translate=(0.1, 0.1), scale=(1, 1.1)),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
if config['resize']:
    transforms_list.append(
        transforms.Resize((128, 128))
    )
train_transform = transforms.Compose(transforms_list)

# Get data
z_train_loader = DataLoader(ZEBRAS(dataset="train", transform=train_transform), batch_size=config["batch_size"], num_workers=4)
h_train_loader = DataLoader(HORSES(dataset="train", transform=train_transform), batch_size=config["batch_size"], num_workers=4)
z_test_loader = DataLoader(ZEBRAS(dataset="test", transform=test_transform), batch_size=config["batch_size"], num_workers=4)
h_test_loader = DataLoader(HORSES(dataset="test", transform=test_transform), batch_size=config["batch_size"], num_workers=4)

# Train model
train_cycle_gan(config, g_h2z, g_z2h, d_h, d_z, z_train_loader, h_train_loader, "project-3")