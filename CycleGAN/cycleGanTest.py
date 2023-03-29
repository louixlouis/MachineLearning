import os
import itertools

from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torchvision.utils import save_image

from model import Generator, Discriminator
from dataLoader import ImageDataset

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    batch_size = 1

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)
    outputs_path = './outputs'
    os.makedirs(outputs_path, exist_ok=True)
    
    transform = [
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    testset = ImageDataset(root='../datasets/vangogh2photo', transforms_=transform, unaligned=True, mode='test')
    test_batches = torch.utils.data.DataLoader(
        dataset = testset,
        batch_size = batch_size,
        shuffle = True,
    )

    # Models.
    generator_AB = Generator().to(device)
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_G_AB_20.tar'))
    generator_AB.load_state_dict(checkpoint['model'])
    generator_BA = Generator().to(device)
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_G_BA_20.tar'))
    generator_BA.load_state_dict(checkpoint['model'])
    
    # Training loop
    with torch.no_grad():
        generator_AB.eval()
        generator_BA.eval()

        for iter, data in enumerate(test_batches):
            image_A = data['A'].to(device)
            image_B = data['B'].to(device)
            
            # Calculate identity loss
            image_B_to_A = generator_BA(image_B)
            image_A_to_B = generator_AB(image_A)
            save_image(torch.cat([image_A, image_A_to_B], dim=0).data, os.path.join(outputs_path, f'fake_sample_AB_{iter+1}.png'), nrow=2, padding=0, normalize=True)
            save_image(torch.cat([image_B, image_B_to_A], dim=0).data, os.path.join(outputs_path, f'fake_sample_BA_{iter+1}.png'), nrow=2, padding=0, normalize=True) 