import os
import itertools

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
    