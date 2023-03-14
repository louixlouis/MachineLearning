import os

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from model import VariationalAutoEncoder

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    
    learning_rate = 0.001
    training_epochs = 30
    batch_size = 128
    
    reconstruction_path = './reconstruction'
    os.makedirs(reconstruction_path, exist_ok=True)

    sample_path = './sample'
    os.makedirs(sample_path, exist_ok=True)

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

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
    
    model = VariationalAutoEncoder().to(device)
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_29.tar'))
    model.load_state_dict(checkpoint['model'])

    total_batch = len(test_batches)

    # Test loop
    with torch.no_grad():
        model.eval()
        for iter, (X_test, Y_label_test) in enumerate(test_batches):
            X_test = X_test.view(X_test.shape[0], -1).to(device)
            mu, log_var = model.encoder(X_test)
            z = model.reparameterize(mu, log_var)
            reconstructed = model.decoder(z)
            reconstructed = torch.cat([X_test.view(-1, 1, 28, 28), reconstructed.view(-1, 1, 28, 28)], dim=3)
            save_image(reconstructed, os.path.join(reconstruction_path, f'{iter+1}.png'))

        # z = torch.randn(batch_size, 10).to(device)
        # print(z)
        # sample_image = model.decoder(z)
        # save_image(sample_image.view(-1, 1, 28, 28), os.path.join(sample_path, f'sample.png'))
        
        x_values = np.linspace(-3, 3, 20)
        y_values = np.linspace(-3, 3, 20)
        canvas = np.empty((28*20, 28*20))
        for i, x_val in enumerate(x_values):
            for j, y_val in enumerate(y_values):
                z = np.array([[x_val, y_val] * 1]).reshape(1, 2)
                reconstructed = model.decoder(torch.Tensor(z))
                canvas[(20-i-1)*28:(20-i)*28, j*28:(j+1)*28] = reconstructed[0].reshape(28, 28)
        plt.figure(figsize=(8, 10))
        plt.imshow(canvas, origin='upper')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show()