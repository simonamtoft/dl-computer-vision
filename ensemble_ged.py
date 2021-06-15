from helpers.performance_metrics import compute_iou
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
    'epochs': 15,
    'batch_size': 384,
    'learning_rate': 1e-4,
    'optimizer': 'adam',
    'loss_func': ['bce_weight', [1]],
    'batch_norm': True,
    'dropout': 0.15,
    'channels': [64, 64],
    'n_convs': 3,
    'padding': 1,
    'step_lr': [True, 1, 0.975],
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


import pickle
models = pickle.load(open("./ensemble_models.pickle", "rb+"))

X, Y = next(iter(data_val))
X = X.to(device)
Y = Y.to(device)
rng = np.random.default_rng()
t1 = 0
t2 = 0
t3 = 0
N = min(len(X), 100)
for i in range(N):
    x = X[i]
    m = rng.choice(4)
    a = rng.choice(4)
    m2 = rng.choice(4)
    a2 = rng.choice(4)

    y = Y[i][a][None]
    y_hat = models[m](x[None]) >= 0.5
    y2 = Y[i][a2][None]
    y_hat2 = models[m2](x[None]) >= 0.5

    t1 += (1 - compute_iou(y_hat, y, 1))/N
    t2 += (1 - compute_iou(y, y2, 1))/N
    t3 += (1 - compute_iou(y_hat, y_hat2, 1))/N
GED = 2*t1 - t2 - t3
print(GED)

# with torch.no_grad():
#     Y_hats = torch.stack([torch.sigmoid(models[i](X.to(device))).detach() for i in range(4)], dim=1).cpu()

# # 
# Y_hats_std = torch.std(Y_hats, axis=1).cpu()
# Y_val_std = torch.std(Y, axis=1).cpu()
# Y_hats_mean = torch.mean(Y_hats, axis=1).cpu()
# Y_val_mean = torch.mean(Y, axis=1).cpu()
# f, ax = plt.subplots(3, 6, figsize=(14, 6))
# for k in range(6):
#     ax[0,k].imshow(X[k, 0].cpu().numpy(), cmap='gray')
#     ax[0,k].set_title('Real data')
#     ax[0,k].axis('off')

#     ax[1,k].imshow(Y_hats_std[k, 0], cmap='hot')
#     ax[1,k].set_title('Ensemble Std')
#     ax[1,k].axis('off')

#     ax[2,k].imshow(Y_val_std[k, 0], cmap='hot')
#     ax[2,k].set_title('Segmentation Std')
#     ax[2,k].axis('off')
# plt.show()
# plt.savefig(f"fig_std.png", transparent=True)
# plt.close()