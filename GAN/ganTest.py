import os

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

    generator = Generator().to(device)
    checkpoint = torch.load(os.path.join(checkpoints_path, 'model_G_4.tar'))
    generator.load_state_dict(checkpoint['model'])

    # Training loop
    with torch.no_grad():
        generator.eval()
        # Train with fake.
        noise = torch.randn(128, 100, 1, 1, device=device)
        fake_image = generator(noise)
        save_image(fake_image, 'smaple.png')
