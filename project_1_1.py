import models as m
import data as d
from helpers import plt_saliency_img
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "model_1"
project_name = "project-1-1"
config_1_1 = {
    'epochs': 20,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'conv_dim': [16, 16, 16, 32, 32, 32, 48, 48, 64],
    'fc_dim': [1024],
    'batch_norm': True,
    'step_lr': [True, 1, 0.8],
}
in_channels = 3


if __name__=="__main__":
    # Define data transforms
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0)),
        transforms.RandomAffine(30, scale=(1, 1.5), shear=3),
    ])

    # Load data
    trainset = d.Hotdog_NotHotdog(data_path='./data/hotdog_nothotdog', train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=config_1_1["batch_size"], shuffle=True, num_workers=1)
    testset = d.Hotdog_NotHotdog(data_path='./data/hotdog_nothotdog', train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=config_1_1["batch_size"], shuffle=False, num_workers=1)

    images, labels = next(iter(test_loader))
    img = images[23]

    # Instantiate model
    model = m.StandardCNN(config=config_1_1).to(device)

    # # train model
    out_dict = m.train(model, config_1_1, project_name, train_loader, test_loader)

    # # pick a single image
    # images, labels = next(iter(test_loader))
    # img = images[23]

    # # create and save salience map
    # plt_saliency_img(img, model, save_img=True)

