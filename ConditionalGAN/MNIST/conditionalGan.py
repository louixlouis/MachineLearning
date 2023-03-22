import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets
from torchvision.utils import save_image

from model import Generator, Discriminator

def save_checkpoint(model, name, opt, epoch, path):   
    torch.save({
        'model':model.state_dict(),     
        'optimizer': opt.state_dict(),  
        'epoch': epoch                  
    }, os.path.join(path, f'model_{name}_{epoch+1}.tar'))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.0002
    training_epochs = 10
    batch_size = 64

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    samples_path = './samples'
    os.makedirs(samples_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5)
    ])

    # trainset = datasets.ImageFolder(root='./data', transform=transform)
    trainset = datasets.MNIST(root='./data', transform=transform)
    training_batches = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = batch_size,
        shuffle = True,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    loss_function = nn.BCELoss()
    optimizerG = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    # num_batch = len(training_batches)

    real_label = torch.ones((batch_size, 1)).to(device)
    fake_label = torch.zeros((batch_size, 1)).to(device)

    fixed_noise_z = torch.randn(100, 100, 1, 1, device=device).to(device)
    
    one_hot = torch.zeros(10, 10).scatter_(1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1)
    fixed_noise_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10).type(torch.LongTensor)
    fixed_noise_y = one_hot(fixed_noise_y).to(device)
    
    fill = torch.zeros([10, 10, 28, 28])
    for i in range(10):
        fill[i, i, :, :] = 1

    # Training loop
    for epoch in range(training_epochs):
        for iter, (X, Y_label) in enumerate(training_batches):
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # Train with real.
            optimizerD.zero_grad()
            X = X.to(device)
            # batch * 10 * 28 * 28
            Y_label = fill(Y_label).to(device)
            
            prediction = discriminator(X, Y_label).view(-1)
            lossD_real = loss_function(prediction, real_label)
            lossD_real.backward()

            # Train with fake.
            noise_z = torch.randn(batch_size, 100, 1, 1, device=device)
            noise_y = (torch.rand(batch_size, 1)*10).type(torch.LongTensor).squeeze()

            # Generate fake image.
            noise_y = one_hot[noise_y].to(device)
            fake_image = generator(noise_z, noise_y)
            noise_y = fill[noise_y]
            prediction = discriminator(fake_image.detach(), noise_y)
            lossD_fake = loss_function(prediction, fake_label)
            lossD_fake.backward()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # Update G network: maximize log(D(G(z)))
            optimizerG.zero_grad()
            prediction = discriminator(fake_image, noise_y)
            lossG = loss_function(prediction, real_label)
            lossG.backward()
            optimizerG.step()

            if (iter+1) % 100 == 0:
                fake_image = generator(fixed_noise_z, fixed_noise_y)
                save_image(fake_image.data, os.path.join(samples_path, f'fake_sample_{epoch+1}_{iter+1}.png'), normalize=True)
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    % (epoch+1, training_epochs, iter+1, len(training_batches), lossD.item(), lossG.item()))
        save_checkpoint(generator, 'G', optimizerG, epoch, checkpoints_path)
        save_checkpoint(discriminator, 'D', optimizerD, epoch, checkpoints_path)