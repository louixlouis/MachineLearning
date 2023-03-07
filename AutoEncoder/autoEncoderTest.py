import os

import torch
import torch.nn as nn

from torchvision import transforms, datasets

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.ReLU(),
        )
    def forward(self, x):
        latent_vec = self.encoder(x)
        reconstructed = self.decoder(latent_vec)
        return latent_vec, reconstructed

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    learning_rate = 0.005
    training_epochs = 10
    batch_size = 64

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

    print(testset.)
    # model = AutoEncoder().to(device)
    # total_batch = len(test_batches)

    # # Test loop
    # for epoch in range(training_epochs):
    #     avg_cost = 0.0
    #     for X, Y_label in test_batches:
    #         X = X.view(-1, 28*28).to(device)
    #         # Y_label = Y_label.to(device)

    #         _, reconstructed = model(X)