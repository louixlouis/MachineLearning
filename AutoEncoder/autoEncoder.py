import os

import torch
import torch.nn as nn

from torchvision import transforms, datasets

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder_layer = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 3*3*32),
            nn.ReLU()
        )
        self.decoder_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
    def forward(self, x):
        out = self.encoder_layer(x)
        latent_vec = self.encoder_fc(out.view(out.shape[0], -1))
        out = self.decoder_fc(latent_vec)
        reconstructed = self.decoder_layer(out.view(out.shape[0], 32, 3, 3))
        return latent_vec, reconstructed

def save_checkpoint(model, opt, epoch, path):   
    torch.save({
        'model':model.state_dict(),     
        'optimizer': opt.state_dict(),  
        'epoch': epoch                  
    }, os.path.join(path, f'model_{epoch}.tar'))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    learning_rate = 0.001
    training_epochs = 20
    batch_size = 256

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5)
    ])

    trainset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    training_batches = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = batch_size,
        shuffle = True,
    )

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_batch = len(training_batches)

    # Training loop
    for epoch in range(training_epochs):
        avg_cost = 0.0
        for X, Y_label in training_batches:
            X = X.to(device)
            # Y_label = Y_label.to(device)

            _, reconstructed = model(X)

            loss = criterion(reconstructed, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_cost += loss / total_batch
        save_checkpoint(model, optimizer, epoch, checkpoints_path)
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))