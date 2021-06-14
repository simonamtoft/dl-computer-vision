import torch
from PIL import Image
from glob import glob
import numpy as np
from torchvision.transforms import ToTensor
import os

class LIDC(torch.utils.data.Dataset):
  def __init__(self, transform, common_transform, dataset='train', annotator=0, data_path="LIDC"):
    """ dataset = 'train', 'val', 'test'
        annotator = 0, 1, 2, 3
    """
    # get image paths
    data_path += f"/LIDC_crops/LIDC_DLCV_version/{dataset}/images"
    self.image_paths = glob(data_path + "/*.png")

    # get segmentation paths
    self.seg_paths = [path.replace(".png", "_l" + str(annotator) + ".png") for path in self.image_paths]
    self.seg_paths = [path.replace("images", "lesions") for path in self.seg_paths]

    self.transform = transform
    self.common_transform = common_transform

  def __len__(self):
    'Returns the total number of samples'
    return len(self.image_paths)

  def __getitem__(self, idx):
    'Generates one sample of data'
    image_path = self.image_paths[idx]
    seg_path = self.seg_paths[idx]
    
    image = Image.open(image_path)
    segmentation = Image.open(seg_path)

    y = self.common_transform(segmentation)
    X = self.common_transform(self.transform(image))
    return X, y


class LIDC_CLDV(torch.utils.data.Dataset):
  def __init__(self, split="train", transform=ToTensor(), data_path="LIDC_crops/LIDC_DLCV_version", annotator=-1):
    self.transform = transform
    data_path = os.path.join(data_path, split)
    self.image_paths = glob(data_path + "/images/*.png")
    if annotator == -1:
      self.lesion_paths = [
        [path.replace("images", "lesions").replace(".png", f"_l{i}.png") for path in self.image_paths]
        for i in range(4)
      ]
    else:
      self.lesion_paths = [
        [path.replace("images", "lesions").replace(".png", f"_l{annotator}.png") for path in self.image_paths]
      ]

  def __len__(self):
      'Returns the total number of samples'
      return len(self.image_paths)

  def __getitem__(self, idx):
      'Generates one sample of data'
      image = Image.open(self.image_paths[idx])
      segmentations = [ToTensor()(Image.open(annotator[idx])) for annotator in self.lesion_paths]
      Y = torch.stack(segmentations)
      X = ToTensor()(image)
      return X, Y