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

def gradient_penalty(discriminator, real, fake, alpha, step, device='cpu'):
    batch_size, channel, height, width = real.shape
    beta = torch.rand((batch_size, 1, 1, 1)).repeat(1, channel, height, width).to(device)
    image_hat = beta*real + (1-beta)*fake.detach()
    image_hat.requires_grad_(True)

    loss = discriminator(image_hat, alpha, step)
    gradient = torch.autograd.grad(outputs=loss, inputs=image_hat, grad_outputs=torch.ones_like(loss), create_graph=True, retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty
    
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
    start_image_size = 4
    data_root = ''

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
    step = int(log2(start_image_size / 4))
    for epochs in epoch_list[step:]:
        alpha = 1e-7
        dataset, loader = get_loader(image_size=4*2**step, batch_size_list=batch_size_list, data_root=data_root)
        print(f'Current image size : {4*2**step} X {4*2**step}')

        for epoch in range(epochs):
            print(f'Epoch [{epoch+1}]/[{epochs}]')
            for i, (real, _) in enumerate(tqdm(loader, leave=True)):
                real = real.to(device)
                cur_batch_size = real.shape[0]
                noise = torch.randn(cur_batch_size, z_dim).to(device)
                fake = generator(noise, alpha, step)
                loss_real = discriminator(real, alpha, step)
                loss_fake = discriminator(fake.detach(), alpha, step)
                gp = gradient_penalty()
        step += 1
if __name__ == '__main__':
    main()