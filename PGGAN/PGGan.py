import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from model import Generator, Discriminator

def save_checkpoint(model, name, opt, epoch, num_blocks, alpha, path):   
    torch.save({
        'model':model.state_dict(),     
        'optimizer': opt.state_dict(),  
        'epoch': epoch,
        'num_blocks': num_blocks,
        'alpha': alpha                  
    }, os.path.join(path, f'model_{name}_{epoch+1}.tar'))

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.00001
    latent_dim = 512
    training_epochs = 40
    start_epoch = 0
    batch_size_list = [16, 16, 16, 8, 4]
    resolution = 1024

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    samples_path = './samples'
    os.makedirs(samples_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.ImageFolder(root='../../dataset/celebaMWclassified', transform=transform)
    train_dataloader = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
    )

    generator = Generator(latent_dim=latent_dim, resolution=resolution).to(device)
    discriminator = Discriminator(latent_dim=latent_dim, resolution=resolution).to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0, 0.999))

    for epoch in range(start_epoch, training_epochs):
        for iter, samples in enumerate(train_dataloader):
            # Train Disriminator
            optimizer_D.zero_grad()
            loss_D = None
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            loss_G = None
            loss_G.backward()
            optimizer_G.step()

        save_checkpoint(
            model=generator, 
            name='G', 
            opt=optimizer_G, 
            epoch=epoch, 
            num_blocks=0, 
            alpha=1, 
            path=checkpoints_path)
        save_checkpoint(
            model=discriminator,
            name='D', 
            opt=optimizer_D, 
            epoch=epoch, 
            num_blocks=0, 
            alpha=1, 
            path=checkpoints_path)

        with torch.no_grad():
            pass