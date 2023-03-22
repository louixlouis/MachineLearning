import os

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torchvision.utils import save_image

from model import Generator

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    batch_size = 10

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)
    outputs_path = './outputs'
    os.makedirs(outputs_path, exist_ok=True)

    generator = Generator().to(device)
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_G_10.tar'))
    generator.load_state_dict(checkpoint['model'])

    one_hot = torch.zeros(10, 10).scatter_(1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1)

    # Training loop
    with torch.no_grad():
        generator.eval()
        # Generate female images.
        noise_z = torch.randn(batch_size, 100, 1, 1, device=device)
        noise_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).type(torch.LongTensor)
        noise_y_G = one_hot[noise_y].to(device)
        fake_image = generator(noise_z, noise_y_G)
        save_image(fake_image, os.path.join(outputs_path, 'smaple.png'), nrow=10, padding=0, normalize=True)