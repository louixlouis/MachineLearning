import os
import random

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms, datasets

from model import AutoEncoder

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
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
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_19.tar'))
    model.load_state_dict(checkpoint['model'])

    #####
    # Test loop
    #####
    with torch.no_grad():
        model.eval()

        result_images = Image.new('L', (28*10, 28*2))
        for i in range(10):
            r = random.randint(0, 999)
            X_test = testset.data[r + 1000*i : r + 1 + 1000*i].view(1,1,28,28).float().to(device)
            _, reconstructed = model(X_test)
            # plt.imshow(X_test.view(28, 28).cpu(), cmap='gist_gray')
            plt.imshow(reconstructed.squeeze().cpu(), cmap='gist_gray')
            plt.show()
            # result_images.paste(transforms.ToPILImage()(X_test[0].cpu()), (28*i, 0))
            # result_images.paste(transforms.ToPILImage()(reconstructed.squeeze()), (28*i, 28))

        result_images.show()