import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from model import *

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.0002
    training_epochs = 20
    batch_size = 64

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    samples_path = './samples'
    os.makedirs(samples_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.ImageFolder(root='../../dataset/celebaMWclassified', transform=transform)
    training_batches = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
    )