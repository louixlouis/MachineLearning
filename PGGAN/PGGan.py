import os

import torch
import torch.nn as nn
from torchvision import transforms, datasets

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
    epoch_list = [5, 15, 25, 35, 40]
    batch_size_list = [16, 16, 16, 8, 4]
    resolution = 1024
    lambda_ = 10
    resume = False

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

    generator = Generator(latent_dim=latent_dim, resolution=resolution).to(device)
    discriminator = Discriminator(latent_dim=latent_dim, resolution=resolution).to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0, 0.999))

    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

    if start_epoch > 0:
        checkpoint = torch.load(checkpoints_path())
        # Load pretrained parameters

    # # The schedule contains [num of epoches for starting each size][batch size for each size][num of epoches]
    # schedule = [[5, 15, 25 ,35, 40],[16, 16, 16, 8, 4],[5, 5, 5, 1, 1]]
    try:
        num = next(iter for iter, epoch in enumerate(epoch_list) if epoch > start_epoch) - 1
        trainset = datasets.ImageFolder(root='../../dataset/celebaMWclassified', transform=transform)
        train_dataloader = torch.utils.data.DataLoader(
            dataset = trainset,
            batch_size = batch_size_list[num],
            shuffle = True,
            # drop_last = True,
        )    
        total_iter = len(trainset) / batch_size_list[num]
        generator.fade_iters = (1 - generator.alpha) / (batch_size_list[num+1] - start_epoch) / (2*total_iter)
        discriminator.fade_iters = (1 - discriminator.alpha) / (batch_size_list[num+1] - start_epoch) / (2*total_iter)
    except:
        trainset = datasets.ImageFolder(root='../../dataset/celebaMWclassified', transform=transform)
        train_dataloader = torch.utils.data.DataLoader(
            dataset = trainset,
            batch_size = batch_size_list[-1],
            shuffle = True,
            # drop_last = True,
        ) 
        total_iter = len(trainset) / batch_size_list[-1]
        if generator.alpha < 1:
            generator.fade_iters = (1 - generator.alpha) / (training_epochs - start_epoch) / (2*total_iter)
            discriminator.fade_iters = (1 - discriminator.alpha) / (training_epochs - start_epoch) / (2*total_iter)

    
    for epoch in range(start_epoch, training_epochs):
        if epoch in epoch_list:
            if pow(2, generator.num_blocks) < resolution:
                pass
            
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