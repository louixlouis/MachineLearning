import os
import shutil

import torchvision

def MNIST():
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    for iter, data in enumerate(trainset):
        image, label = data
        data_path = f'./data/train_data/{label}'
        os.makedirs(data_path, exist_ok=True)
        image.save(os.path.join(data_path, f'{label}_{iter+1}.png'))
        print(f'Image num [{iter+1:>6d}] is saved.')
    shutil.rmtree('./data/MNIST')
if __name__=='__main__':
    MNIST()