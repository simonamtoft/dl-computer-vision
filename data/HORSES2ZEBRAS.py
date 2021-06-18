import glob
import torch
from PIL import Image


class HORSES(torch.utils.data.Dataset):
  def __init__(self, transform, dataset="train", data_path="horse2zebra/"):
    """ dataset = "train", "test" """
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
  def __init__(self, transform, dataset="train", data_path="horse2zebra/"):
    """ dataset = "train", "test" """
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
