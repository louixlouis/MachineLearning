import os
import itertools

from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torchvision.utils import save_image

from model import Generator, Discriminator

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device " {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    batch_size = 16

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)
    outputs_path = './outputs'
    os.makedirs(outputs_path, exist_ok=True)
    
    transform = transforms.Compose([
        # transforms.Resize(int(128*1.12), Image.BICUBIC),
        transforms.Resize(128),
        transforms.RandomCrop(128),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = datasets.ImageFolder(root='./data', transform=transform)
    test_batches = torch.utils.data.DataLoader(
        dataset = testset,
        batch_size = batch_size,
        shuffle = True,
    )

    # Models.
    generator_AB = Generator().to(device)
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_G_AB_10.tar'))
    generator_AB.load_state_dict(checkpoint['model'])
    generator_BA = Generator().to(device)
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_G_BA_10.tar'))
    generator_BA.load_state_dict(checkpoint['model'])
    
    # Training loop
    with torch.no_grad():
        generator_AB.eval()
        generator_BA.eval()
        for epoch in range(test_batches):
            for iter, (X, Y) in enumerate(test_batches):
                image_A = X['A'].to(device)
                image_B = X['B'].to(device)
                
                # Calculate identity loss
                image_B_to_A = generator_BA(image_B)
                image_A_to_B = generator_AB(image_A)

                if (iter+1) % 100 == 0:
                    # fake_image = generator(fixed_noise_z, fixed_noise_y)
                    # save_image(fake_image.data, os.path.join(samples_path, f'fake_sample_{epoch+1}_{iter+1}.png'), nrow=10, padding=0, normalize=True)
                    print(f'[{epoch+1:>2}/{training_epochs:>2}]\nLoss_D: {total_loss_G.item()}\tLoss_D_A: {loss_D_A.item()}\tLoss_D_B: {loss_D_B.item()}') 