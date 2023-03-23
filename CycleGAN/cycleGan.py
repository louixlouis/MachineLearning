import os
import itertools

from PIL import Image

import torch
import torch.nn as nn

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
    print(f'Device " {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.0002
    training_epochs = 20
    batch_size = 16

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize(int(128*1.12), Image.BICUBIC),
        # transforms.Resize(128),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.ImageFolder(root='./data', transform=transform)
    training_batches = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = batch_size,
        shuffle = True,
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

    # Learning rate schedulers.
    real_label = torch.ones((batch_size, 1)).to(device)
    fake_label = torch.zeros((batch_size, 1)).to(device)
    
    # Training loop
    for epoch in range(training_epochs):
        for iter, (X, Y) in enumerate(training_batches):
            image_A = X['A'].to(device)
            image_B = X['B'].to(device)

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
            pred_fake = discriminator_B(image_A_to_B.detach())
            loss_fake = loss_gan(pred_fake, fake_label)

            # Total discriminator A loss
            loss_D_B = 0.5*(loss_real + loss_fake)
            loss_D_B.backward()
            optimizerD_A.step()

            if (iter+1) % 100 == 0:
                # fake_image = generator(fixed_noise_z, fixed_noise_y)
                # save_image(fake_image.data, os.path.join(samples_path, f'fake_sample_{epoch+1}_{iter+1}.png'), nrow=10, padding=0, normalize=True)
                print(f'[{epoch+1:>2}/{training_epochs:>2}]\nLoss_D: {total_loss_G.item()}\tLoss_D_A: {loss_D_A.item()}\tLoss_D_B: {loss_D_B.item()}') 
        # It need to save both models?
        save_checkpoint(generator_AB, 'G_AB', optimizerG, epoch, checkpoints_path)
        save_checkpoint(generator_BA, 'G_BA', optimizerG, epoch, checkpoints_path)
        save_checkpoint(discriminator_A, 'D_A', optimizerD_A, epoch, checkpoints_path)
        save_checkpoint(discriminator_B, 'D_B', optimizerD_B, epoch, checkpoints_path)