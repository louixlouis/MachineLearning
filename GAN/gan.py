import os

import torch
import torch.nn as nn

from torchvision import transforms, datasets

from model import Generator, Discriminator

def save_checkpoint(model, name, opt, epoch, path):   
    torch.save({
        'model':model.state_dict(),     
        'optimizer': opt.state_dict(),  
        'epoch': epoch                  
    }, os.path.join(path, f'model_{name}_{epoch}.tar'))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device " {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.0001
    training_epochs = 5
    batch_size = 128

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.ImageFolder(root='./data', transform=transform)
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

    real_label = 1
    fake_label = 0
    # Training loop
    for epoch in range(training_epochs):
        for iter, (X, Y) in enumerate(training_batches):
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # Train with real.
            optimizerD.zero_grad()
            X = X.to(device)
            label = torch.full((X.shape[0],), real_label, dtype=torch.float, device=device)
            prediction = discriminator(X).view(-1)
            lossD_real = loss_function(prediction, label)
            lossD_real.backward()

            # Train with fake.
            noise = torch.randn(X.shape[0], 100, 1, 1, device=device)
            # Generate fake image.
            fake_image = generator(noise)
            label.fill_(fake_label)
            prediction = discriminator(fake_image.detach()).view(-1)
            lossD_fake = loss_function(prediction, label)
            lossD_fake.backward()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # Update G network: maximize log(D(G(z)))
            optimizerG.zero_grad()
            label.fill_(real_label)
            prediction = discriminator(fake_image).view(-1)
            lossG = loss_function(prediction, label)
            lossG.backward()
            optimizerG.step()

            if iter % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    % (epoch, training_epochs, iter, len(training_batches), lossD.item(), lossG.item()))
        save_checkpoint(generator, 'G', optimizerG, epoch, checkpoints_path)
        save_checkpoint(discriminator, 'D', optimizerD, epoch, checkpoints_path)