import os
import csv
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class HORSES(torch.utils.data.Dataset):
  def __init__(self, transform, dataset = "train", data_path="horse2zebra/"):
    """ dataset = "train","test"
        
    """
    self.dataset = dataset
    data_path0 = data_path + dataset + "/A"
    self.image_paths = glob.glob(data_path0 + "/*.jpg")
    self.transform = transform

  def __len__(self):
    'Returns the total number of samples'
    return len(self.image_paths)

  def __getitem__(self, idx):
    'Generates one sample of data'
    image_path = self.image_paths[idx]
    image = Image.open(image_path).convert('RGB')
    X = self.transform(image)
    return X
  
class ZEBRAS(torch.utils.data.Dataset):
  def __init__(self, transform, dataset = "train", data_path="horse2zebra/"):
    """ dataset = "train","test"
        
    """
    self.dataset = dataset
    data_path0 = data_path + dataset + "/B"
    self.image_paths = glob.glob(data_path0 + "/*.jpg")
    self.transform = transform

  def __len__(self):
    'Returns the total number of samples'
    return len(self.image_paths)

  def __getitem__(self, idx):
    'Generates one sample of data'
    image_path = self.image_paths[idx]
    image = Image.open(image_path).convert('RGB')
    X = self.transform(image)
    return X
  
#batch_size = 1

#transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])

#trainset_H = HORSES(dataset = "train", transform=transform)
#testset_H = HORSES(dataset = "test", transform=transform)
#trainset_Z = ZEBRAS(dataset = "train", transform=transform)
#testset_Z = ZEBRAS(dataset = "test", transform=transform)
