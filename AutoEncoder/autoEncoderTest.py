import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms, datasets

from model import AutoEncoder

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.001
    training_epochs = 10
    batch_size = 256

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    #####
    # Load test dataset.
    #####
    testset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_batches = torch.utils.data.DataLoader(
        dataset = testset,
        batch_size = batch_size,
        shuffle = True,
    )

    #####
    # Define model.
    #####
    model = AutoEncoder().to(device)

    #####
    # Load checkpoint
    #####
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_29.tar'))
    model.load_state_dict(checkpoint['model'])

    #####
    # Test loop
    #####
    '''
    figsize = (width, height) 
    default : 6.4 X 4.8 in inch
    '''
    num = 10
    plt.figure(figsize=(16, 4.5))
    labels = testset.targets.numpy()
    label_index = {i:np.where(labels == i)[0][0] for i in range(num)}
    for i in range(num):
        '''
        subplot(row, col, ?)
        '''
        ax = plt.subplot(2, num, i +1)
        image = testset[label_index[i]][0].unsqueeze(dim=0).to(device)
        with torch.no_grad():
            model.eval()
            _, reconstructed = model(image)
            plt.imshow(image.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if i == num // 2:
            ax.set_title('Original images')    
        ax = plt.subplot(2, num, i + 1 + num)
        plt.imshow(reconstructed.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == num//2:
            ax.set_title('Reconstructed images')
    plt.show()