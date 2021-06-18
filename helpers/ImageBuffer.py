import random
import torch
from torch.autograd import Variable

class ImageBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.images = []

    def push_and_pop(self, batch):
        ret = []
        for image in batch.data:
            image = torch.unsqueeze(image, 0)
            if len(self.images) < self.max_size:
                self.images.append(image)
                ret.append(image)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    ret.append(self.images[i].clone())
                    self.images[i] = image
                else:
                    ret.append(image)
        return Variable(torch.cat(ret)) 