import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = Image.open(self.files_A[index % len(self.files_A)])
        # print(f'A {item_A.mode}')
        item_A = self.transform(item_A)
        
        if self.unaligned:
            item_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            # print(f'B {item_B.mode}')
            item_B = self.transform(item_B)
        else:
            item_B = Image.open(self.files_B[index % len(self.files_B)])
            item_B = self.transform(item_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))