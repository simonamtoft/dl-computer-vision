import torch
import os
import glob
from PIL import Image
from torch.utils.data import DataLoader


class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


def load_hotdog(train_transform, test_transform, config, path='./', num_workers=2):
    trainset = Hotdog_NotHotdog(data_path=path, train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    testset = Hotdog_NotHotdog(data_path=path, train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
