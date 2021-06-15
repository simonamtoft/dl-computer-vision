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
from models import loss_func, UNet, train_medical, train_ensemble, train_anno_ensemble
from data import LIDC, LIDC_CLDV

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'epochs': 20,
    'batch_size': 384,
    'learning_rate': 1e-4,
    'optimizer': 'adam',
    'loss_func': ['bce_weight', [1]],
    'batch_norm': True,
    'dropout': 0.15,
    'channels': [64, 64],
    'n_convs': 3,
    'padding': 1,
    'step_lr': [True, 1, 0.95],
}

if not os.path.exists('./LIDC_crops'):
    from zipfile import ZipFile
    import gdown
    url = 'https://drive.google.com/uc?id=1W6zB9UH4j_iHxaUcYZSdiEkzcJN77lSA'
    gdown.download(url, './LIDC_CLDV.zip', quiet=False)
    with ZipFile('LIDC_CLDV.zip', 'r') as zip:
        zip.extractall()


data_tr =  DataLoader(LIDC_CLDV(split="train", annotator=-1, transform="", common_transform=""), batch_size=config["batch_size"], shuffle=True,  num_workers=4)
data_val = DataLoader(LIDC_CLDV(split="val",   annotator=-1, transform="", common_transform=""), batch_size=config["batch_size"], shuffle=False, num_workers=4)

#model = UNet(config=config).to(device)
#train_medical(model, config, data_tr, data_val, "test", True, True)
models = train_anno_ensemble(config, data_tr, data_val, "test", True, True)


import pickle
pickle.dump(models, open("./ensemble_models.pickle", "wb+"))
#model = pickle.load(open("./model.pickle", "rb+"))