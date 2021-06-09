import torch
import os
import glob
from PIL import Image
import torchvision.transforms as transforms


class SVHNCorners(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='SVHN'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')        
        self.image_paths = glob.glob(data_path + '/*.png')
        self.opened_idx = -1
        self.crops = None
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths) * 4 # 4 corners

    def __getitem__(self, idx):
        'Generates one sample of data'
        if not self.opened_idx == idx//4:
          image_path = self.image_paths[idx//4]
          image = Image.open(image_path)
          min = np.min(image.size)
          if min < 64: # Avoid overlap
            image = transforms.Resize(64)(image)
          self.crops = transforms.FiveCrop(32)(image)
        
        crop = self.crops[idx % 4]
        y = 10
        X = self.transform(crop)
        return X, y