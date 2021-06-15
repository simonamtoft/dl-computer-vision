import torch
from PIL import Image
from glob import glob
import numpy as np
import random
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
import os

class LIDC(torch.utils.data.Dataset):
  def __init__(self, transform, common_transform, split='train', annotator=0, data_path="LIDC"):
    """ dataset = 'train', 'val', 'test'
        annotator = 0, 1, 2, 3
    """
    # get image paths
    data_path += f"{split}/images"
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
  def __init__(self, transform, common_transform, split="train", data_path="LIDC_crops/LIDC_DLCV_version", annotator=-1):
    self.transform = transform
    self.common_transform = common_transform
    self.split = split

    # Get paths
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
      image = ToTensor()(Image.open(self.image_paths[idx]))
      segmentations = [ToTensor()(Image.open(annotator[idx])) for annotator in self.lesion_paths]
      if self.split == "train":
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            segmentations = [TF.rotate(segmentation, angle) for segmentation in segmentations]
        if random.random() > 0.5:
            translate1 = 0.3
            translate2 = 0.3
            max_dx = translate1 * 128
            max_dy = translate2 * 128
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
            image = TF.affine(image,angle = 0, translate=translations, scale = 1, shear = 0)
            segmentations = [TF.affine(segmentation,angle = 0, translate=translations, scale = 1, shear = 0) for segmentation in segmentations]


      Y = torch.stack(segmentations)
      X = image
      return X, Y
