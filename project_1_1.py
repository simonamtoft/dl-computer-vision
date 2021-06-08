import models as m
import data as d
from helpers import plt_saliency_img
import torch
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__=="__main__":
    config = m.config
    in_channels = 3

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
    train_loader, test_loader = d.load_hotdog(train_transform, test_transform, config)

    # Instantiate model
    model = m.StandardCNN(3, config).to(device)

    # train model
    out_dict = m.train(model, config, m.config.project_name, train_loader, test_loader)

    # pick a single image
    images, labels = next(iter(test_loader))
    img = images[23]

    # create and save salience map
    plt_saliency_img(img, model, save_img=True)

