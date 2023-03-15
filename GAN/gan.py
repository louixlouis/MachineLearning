import os

import torch
import torch.nn as nn

from torchvision import transforms, datasets

from model import Generator, Discriminator
def save_checkpoint(model, opt, epoch, path):   
    torch.save({
        'model':model.state_dict(),     
        'optimizer': opt.state_dict(),  
        'epoch': epoch                  
    }, os.path.join(path, f'model_{epoch}.tar'))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device " {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.001
    training_epochs = 30
    batch_size = 128

    checkpoins_path = './checkpoints'
    os.makedirs(checkpoins_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.DatasetFolder()
    training_batches = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = batch_size,
        shuffle = True,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    loss_function = nn.BCELoss()
    optimizerG = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    total_batch = len(training_batches)

    for epoch in range(training_epochs):
        for i, (X, Y) in enumerate(training_batches):
            # Train with real.
            X = X.to(device)
            prediction = discriminator(X)
            lossD_real = loss_function(prediction, Y)

            discriminator.zero_grad()
            lossD_real.backward()

            # Train with fake.