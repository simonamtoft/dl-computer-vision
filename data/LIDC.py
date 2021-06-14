import torch
from PIL import Image
from glob import glob
import numpy as np


class LIDC(torch.utils.data.Dataset):
  def __init__(self, transform, common_transform, dataset='train', annotator=0, data_path="LIDC-IDRI"):
    """ dataset = 'train', 'val', 'test'
        annotator = 0, 1, 2, 3
    """
    annotator = str(annotator)

    data_path += "/LIDC_crops/LIDC_DLCV_version/train/images"
    self.image_paths = glob(data_path + "/*.png")

    self.seg_paths = [path.replace(".png", "_l" + annotator + ".png") for path in self.image_paths]
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

    y = self.common_transform(np.asarray(segmentation))
    X = self.common_transform(self.transform(image))
    return X, y