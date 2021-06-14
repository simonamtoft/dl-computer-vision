# Download data if it does not exist
import sys
import os
import wandb
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, ConcatDataset
from models import loss_func, UNet, train
from data import LIDC, LIDC_CLDV

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'epochs': 20,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'loss_func': 'bce',
    'batch_norm': False,
    'dropout': 0,
    'channels': [64, 64],
    'n_convs': 3,
    'step_lr': [True, 1, 0.8],
}

if not os.path.exists('./LIDC_crops'):
    from zipfile import ZipFile
    import gdown
    url = 'https://drive.google.com/uc?id=1W6zB9UH4j_iHxaUcYZSdiEkzcJN77lSA'
    gdown.download(url, './LIDC_CLDV.zip', quiet=False)
    with ZipFile('LIDC_CLDV.zip', 'r') as zip:
        zip.extractall()


data_tr = DataLoader(LIDC_CLDV(split="train", annotator=0), batch_size=config["batch_size"], shuffle=True, num_workers=2)
data_val = DataLoader(LIDC_CLDV(split="val", annotator=0), batch_size=config["batch_size"], shuffle=False, num_workers=2)

model = UNet(config=config).to(device)


def train(model, opt, loss_fn, epochs, data_tr, data_val):
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            # set parameter gradients to zero
            opt.zero_grad()
            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(F.pad(Y_pred, (3,3,3,3)), Y_batch[:,0,:,:])  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(data_tr)
        print(' - loss: %f' % avg_loss)

train(model, torch.optim.Adam(model.parameters(), config["loss_func"]), loss_func(config["loss_func"]), config["epochs"], data_tr, data_val)