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
    'batch_size': 2,
    'epochs': 50,
    'lr_d': 4*1e-4,
    'lr_g': 2*1e-4,
    'g_loss_weight': [1, 10, 5],
    'n_features': 128,
    'n_blocks': 12,
    'relu_val': 0.2,
    'img_loss': ['l2', 'l1'], # cycle, identity 
    'buffer_size': 50,
}

# Instantiate models
g_h2z = Generator(config).to(device)
g_z2h = Generator(config).to(device)
d_h = Discriminator(config).to(device)
d_z = Discriminator(config).to(device)

# Get data
test_transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])

train_transform = transforms.Compose([
                    transforms.Pad(50, padding_mode='reflect'),
                    transforms.RandomAffine(degrees=5,translate=(0.1,0.1),scale=(1,1.05)),
                    transforms.Resize((128,128)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor()])


z_train_loader = DataLoader(ZEBRAS(dataset="train", transform=train_transform), batch_size=config["batch_size"], num_workers=4)
h_train_loader = DataLoader(HORSES(dataset="train", transform=train_transform), batch_size=config["batch_size"], num_workers=4)
z_test_loader = DataLoader(ZEBRAS(dataset="test", transform=test_transform), batch_size=config["batch_size"], num_workers=4)
h_test_loader = DataLoader(HORSES(dataset="test", transform=test_transform), batch_size=config["batch_size"], num_workers=4)

# Train model
train_cycle_gan(config, g_h2z, g_z2h, d_h, d_z, z_train_loader, h_train_loader, "project-3")