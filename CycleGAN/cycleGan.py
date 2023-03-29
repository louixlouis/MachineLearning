import os
import itertools

from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torchvision.utils import save_image

from model import Generator, Discriminator
from dataLoader import ImageDataset
from utils import *

def save_checkpoint(model, name, opt, epoch, path):   
    torch.save({
        'model':model.state_dict(),     
        'optimizer': opt.state_dict(),  
        'epoch': epoch                  
    }, os.path.join(path, f'model_{name}_{epoch+1}.tar'))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device " {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.0002
    training_epochs = 20
    batch_size = 1

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)
    samples_path = './samples'
    os.makedirs(samples_path, exist_ok=True)
    
    transform = [
        transforms.Resize(int(128*1.12), Image.BICUBIC),
        # transforms.Resize(128),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    trainset = ImageDataset(root='../datasets/vangogh2photo', transforms_=transform, unaligned=True)
    training_batches = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = batch_size,
        shuffle = True,
        drop_last=True,
    )
    # Models.
    generator_AB = Generator().to(device)
    generator_BA = Generator().to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)

    # Loss functions.
    loss_gan = nn.MSELoss()
    loss_cycle = nn.L1Loss()
    loss_identity = nn.L1Loss()
    
    # Optimizer.
    optimizerG = torch.optim.Adam(itertools.chain(generator_AB.parameters(), generator_BA.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    optimizerD_A = torch.optim.Adam(discriminator_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerD_B = torch.optim.Adam(discriminator_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Scheduler.
    '''
    last_epoch: default(-1). initial_lr=>lr
    verbose: default(False). Do not print updated message
    '''
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizerG, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)
    schedulerD_A = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizerD_A, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)
    schedulerD_B = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizerD_B, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

    # Learning rate schedulers.
    real_label = torch.ones((batch_size, 1)).to(device)
    fake_label = torch.zeros((batch_size, 1)).to(device)
    
    # Buffers.
    image_A_to_B_buffer = ReplayBuffer()
    image_B_to_A_buffer = ReplayBuffer()
    
    # Training loop
    for epoch in range(training_epochs):
        for iter, data in enumerate(training_batches):
            image_A = data['A'].to(device)
            image_B = data['B'].to(device)

            # Train Generators.
            optimizerG.zero_grad()
            
            # Calculate identity loss
            image_A_to_A = generator_BA(image_A)
            loss_identity_A = loss_identity(image_A_to_A, image_A)
            image_B_to_B = generator_AB(image_B)
            loss_identity_B = loss_identity(image_B_to_B, image_B)

            # Calculate cycle loss
            image_A_to_B = generator_AB(image_A)
            reconstructed_A = generator_BA(image_A_to_B)
            loss_cycle_ABA = loss_cycle(reconstructed_A, image_A)
            image_B_to_A = generator_BA(image_B)
            reconstructed_B = generator_AB(image_B_to_A)
            loss_cycle_BAB = loss_cycle(reconstructed_B, image_B)

            # Calculate GAN loss
            pred_B = discriminator_B(image_A_to_B)
            loss_gan_A_to_B = loss_gan(pred_B, real_label)
            pred_A = discriminator_A(image_B_to_A)
            loss_gan_B_to_A = loss_gan(pred_A, real_label)

            total_loss_G = 5.0*(loss_identity_A + loss_identity_B) + 10.0*(loss_cycle_ABA + loss_cycle_BAB) + loss_gan_A_to_B + loss_gan_B_to_A
            total_loss_G.backward()
            optimizerG.step()

            # Train Discriminator A.
            optimizerD_A.zero_grad()
            
            # Train with real
            pred_real = discriminator_A(image_A)
            loss_real = loss_gan(pred_real, real_label)
            
            # Train with fake.
            image_B_to_A = image_B_to_A_buffer.push_and_pop(image_B_to_A)
            pred_fake = discriminator_A(image_B_to_A.detach())
            loss_fake = loss_gan(pred_fake, fake_label)

            # Total discriminator A loss
            loss_D_A = 0.5*(loss_real + loss_fake)
            loss_D_A.backward()
            optimizerD_A.step()

            # Train Discriminator B.
            optimizerD_B.zero_grad()

            # Train with real
            pred_real = discriminator_B(image_B)
            loss_real = loss_gan(pred_real, real_label)

            # Train with fake
            image_A_to_B = image_A_to_B_buffer.push_and_pop(image_A_to_B)
            pred_fake = discriminator_B(image_A_to_B.detach())
            loss_fake = loss_gan(pred_fake, fake_label)

            # Total discriminator A loss
            loss_D_B = 0.5*(loss_real + loss_fake)
            loss_D_B.backward()
            optimizerD_B.step()

            if (iter+1) % 100 == 0:
                # fake_image = generator(fixed_noise_z, fixed_noise_y)
                save_image(torch.cat([image_A, image_A_to_B[-1].unsqueeze(dim=0)], dim=0).data, os.path.join(samples_path, f'fake_sample_AB_{iter+1}.png'), nrow=batch_size, padding=0, normalize=True)
                save_image(torch.cat([image_B, image_B_to_A[-1].unsqueeze(dim=0)], dim=0).data, os.path.join(samples_path, f'fake_sample_BA_{iter+1}.png'), nrow=batch_size, padding=0, normalize=True)
                # save_image(image_B_to_A.data, os.path.join(samples_path, f'fake_sample_BA_{iter+1}.png'), nrow=8, padding=0, normalize=True)
                print(f'[{epoch+1:>2d}/{training_epochs:>2d}][{iter+1:>4d}/{len(training_batches):>4d}]\nLoss_G  : {total_loss_G.item():>2.4f}\nLoss_D_A: {loss_D_A.item():>2.4f}\nLoss_D_B: {loss_D_B.item():>2.4f}') 
        # It need to save both models?
        save_checkpoint(generator_AB, 'G_AB', optimizerG, epoch, checkpoints_path)
        save_checkpoint(generator_BA, 'G_BA', optimizerG, epoch, checkpoints_path)
        save_checkpoint(discriminator_A, 'D_A', optimizerD_A, epoch, checkpoints_path)
        save_checkpoint(discriminator_B, 'D_B', optimizerD_B, epoch, checkpoints_path)
        
        # Update learning rate.
        print(f'lr_G  : {optimizerG.param_groups[0]["lr"]:>.6f}')
        print(f'lr_D_A: {optimizerD_A.param_groups[0]["lr"]:>.6f}')
        print(f'lr_D_B: {optimizerD_B.param_groups[0]["lr"]:>.6f}')
        schedulerG.step()
        schedulerD_A.step()
        schedulerD_B.step()

import numpy as np 
