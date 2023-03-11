import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.decomposition import PCA

import torch
import torch.nn as nn

from torchvision import transforms, datasets

from model import AutoEncoder

#####
# Define PCA
#####
pca = PCA(n_components=2)

#####
# Define model.
#####
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {device}')
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
model = AutoEncoder().to(device)

def on_click(event):
    x, y = event.xdata, event.ydata
    if x != None and y != None:
        pass
        # latent_vector = pca.inverse_transform([x, y])
        # latent_vector = torch.from_numpy(latent_vector).float().to(device)
        
        # reconstructed = model.decoder(latent_vector.unsqueeze(dim=0))
        # plt.imshow(reconstructed.cpu().squeeze().numpy(), cmap='gist_gray')

def plot_latent(X, Y):
    pca.fit(X)
    X = pca.transform(X)
    pca_data_df = np.vstack((X.T, Y)).T
    pca_data_df = pd.DataFrame(data=pca_data_df, columns=("x", "y", "label"))
    
    # figure = plt.figure(figsize=(18, 6))
    fig = sb.FacetGrid(pca_data_df, hue='label', height=8, aspect=1).map(plt.scatter, 'x', 'y').add_legend()
    # ax.figure.canvas.mpl_connect('motion_notify_event', on_click)
    fig.canvas.mpl_connect('motion_notify_event', on_click)
    plt.show()

if __name__ == '__main__':
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
    # Load checkpoint
    #####
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_29.tar'))
    model.load_state_dict(checkpoint['model'])

    #####
    # Test loop
    #####
    total_latent_vector = np.array([]).reshape(-1, 4)
    # total_X_test = np.array([])
    total_Y_test = np.array([])
    with torch.no_grad():
        model.eval()
        for X_test, Y_test in test_batches:
            X_test = X_test.to(device)
            latent_vector, _ = model(X_test)
            latent_vector = latent_vector.cpu().numpy()
            total_latent_vector = np.concatenate((total_latent_vector, latent_vector), axis=0)
            total_Y_test = np.concatenate((total_Y_test, Y_test), axis=0)

        plot_latent(total_latent_vector, total_Y_test)
