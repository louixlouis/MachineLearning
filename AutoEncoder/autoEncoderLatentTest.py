import os
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    # Define PCA
    #####
    pca = PCA(n_components=2)

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
    plt.figure(figsize=(8, 4))
    total_latent_vector = np.array([]).reshape(-1, 2)
    total_Y_test = np.array([])
    with torch.no_grad():
        model.eval()
        for X_test, Y_test in test_batches:
            X_test = X_test.to(device)
            latent_vector, _ = model(X_test)
            latent_vector = latent_vector.cpu()
            pca.fit(latent_vector)
            latent_vector = pca.transform(latent_vector)
            total_latent_vector = np.concatenate((total_latent_vector, latent_vector), axis=0)
            total_Y_test = np.concatenate((total_Y_test, Y_test), axis=0)
    plt.subplot(121)
    plt.scatter(total_latent_vector[:5000, 0], total_latent_vector[:5000, 1], c=total_Y_test[:5000], s=8, cmap='tab10')
    plt.show()