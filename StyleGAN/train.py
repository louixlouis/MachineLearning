import os
from math import log2

import numpy as np 
# import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Generator, Discriminator

def get_loader(image_size, batch_size_list, data_root):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = batch_size_list[int(log2(image_size/4))]
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader

def check_loader():
    pass

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    
    learning_rate = 1e-3
    batch_size_list = [256, 256, 128, 64, 32, 16]
    img_channels = 3
    z_dim = 512
    w_dim = 512
    in_channels = 512
    lambda_gp = 10
    epoch_list = [30] * len(batch_size_list)
    factors = [1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

    # Define models.
    generator = Generator(z_dim=z_dim, w_dim=w_dim, in_channels=in_channels, factors=factors, img_channels=img_channels).to(device)
    discriminator = Discriminator(in_channels=in_channels, factors=factors, img_channels=img_channels).to(device)
    optimizer_g = torch.optim.Adam(
        [
            {'params':[param for name, param in generator.named_parameters() if 'mapping' not in name]},
            {'params': generator.mapping.parameters(), 'lr':1e-5}],
        lr=learning_rate,
        betas=(0.0, 0.99))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.99))

    generator.train()
    discriminator.train()

if __name__ == '__main__':
    main()