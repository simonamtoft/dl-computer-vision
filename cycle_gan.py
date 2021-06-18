# Project 3: Convert Horses to Zebras with Cycle GANs
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data import HORSES, ZEBRAS
from models import Discriminator, Generator
from training import train_cycle_gan

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define name of project on weight and biases site
project_name = "project-3"

# Set config of training run
config = {
    'batch_size': 1,
    'epochs': 10,
    'lr_d': 4*1e-4,
    'lr_g': 1*1e-4,
    'n_features': 64,
    'n_blocks': 6,
    'relu_val': 0.2,
    'img_loss': 'l2',
    'g_loss_weight': [1, 10, 5]
}

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor()
])

# Load zebra and horse image data
trainset_z = ZEBRAS(dataset="train", transform=transform)
testset_z = ZEBRAS(dataset= "test", transform=transform)
trainset_h = HORSES(dataset = "train", transform=transform)
testset_h = HORSES(dataset = "test", transform=transform)

# Setup DataLoaders
zebra_loader = DataLoader(trainset_z, batch_size=config['batch_size'], shuffle=True)
horse_loader = DataLoader(trainset_h, batch_size=config['batch_size'], shuffle=True)

# Instantiate Cycle GAN network
d_h = Discriminator(config).to(device)
d_z = Discriminator(config).to(device)
g_h2z = Generator(config).to(device)
g_z2h = Generator(config).to(device)

# Train network
train_cycle_gan(config, g_h2z, g_z2h, d_h, d_z, zebra_loader, horse_loader, project_name)
