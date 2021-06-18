import os
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data.dataloader import DataLoader

from models import Generator, Discriminator
from training import train_cycle_gan
from data import HORSES, ZEBRAS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('./horse2zebra'):
    import gdown
    from zipfile import ZipFile
    url = 'https://drive.google.com/uc?id=1PBHjsYy6cX9PFuH72SWMPSTxHrL6RPhN'
    gdown.download(url, './horse2zebra.zip', quiet=False)
    with ZipFile('horse2zebra.zip', 'r') as zipObj:
        zipObj.extractall()

config = {
    'batch_size': 16,
    'epochs': 10,
    'lr_d': 2*1e-4,
    'lr_g': 1*1e-4,
    'g_loss_weight': [1, 10, 5],
    'n_features': 64,
    'n_blocks': 6,
    'relu_val': 0.2,
    'img_loss': 'l2',
}

# Instantiate models
g_h2z = Generator(config).to(device)
g_z2h = Generator(config).to(device)
d_h = Discriminator(config).to(device)
d_z = Discriminator(config).to(device)

# Get data
transforms = Compose([
    ToTensor(),
    Resize((128, 128))
])
z_train_loader = DataLoader(ZEBRAS(dataset="train", transform=transforms), batch_size=config["batch_size"], num_workers=4)
h_train_loader = DataLoader(HORSES(dataset="train", transform=transforms), batch_size=config["batch_size"], num_workers=4)
z_test_loader = DataLoader(ZEBRAS(dataset="test", transform=transforms), batch_size=config["batch_size"], num_workers=4)
h_test_loader = DataLoader(HORSES(dataset="test", transform=transforms), batch_size=config["batch_size"], num_workers=4)

# Train model
train_cycle_gan(config, g_h2z, g_z2h, d_h, d_z, z_train_loader, h_train_loader, "project-3")
