import torch
from PIL import Image
from glob import glob
import numpy as np


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

    y = self.common_transform(np.asarray(segmentation))
    X = self.common_transform(self.transform(image))
    return X, y