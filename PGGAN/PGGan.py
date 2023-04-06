import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

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
    growing_time_list = [5, 5, 5, 1, 1]     # 뭔가 이상함.
    resolution = 1024
    lambda_ = 10

    data_root = '../datasets/celeba_hq'

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)
    checkpoint_G_path = os.path.join(checkpoints_path, '')
    checkpoint_D_path = os.path.join(checkpoints_path, '')

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

    fixed_latent_z = torch.randn(16, latent_dim, 1, 1, device=device)

    if start_epoch > 0:
        checkpoint_G = torch.load(checkpoint_G_path)
        generator.load_state_dict(checkpoint_G['model'])
        generator.num_blocks = checkpoint_G['num_blocks']
        generator.alpha = checkpoint_G['alpha']
        optimizer_G.load_state_dict(checkpoint_G['optimizer'])
        
        checkpoint_D = torch.load(checkpoint_D_path)
        discriminator.load_state_dict(checkpoint_D['model'])
        discriminator.num_blocks = checkpoint_D['num_blocks']
        discriminator.alpha = checkpoint_D['alpha']
        optimizer_D.load_state_dict(checkpoint_D['optimizer'])

    try:
        num = next(iter for iter, epoch in enumerate(epoch_list) if epoch > start_epoch) - 1
        trainset = datasets.ImageFolder(root=data_root, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(
            dataset = trainset,
            batch_size = batch_size_list[num],
            shuffle = True,
            drop_last = True,
        )    
        total_iter = len(trainset) / batch_size_list[num]
        generator.alpha_step = (1 - generator.alpha) / (batch_size_list[num+1] - start_epoch) / (2*total_iter)
        discriminator.alpha_step = (1 - discriminator.alpha) / (batch_size_list[num+1] - start_epoch) / (2*total_iter)
    except:
        trainset = datasets.ImageFolder(root=data_root, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(
            dataset = trainset,
            batch_size = batch_size_list[-1],
            shuffle = True,
            drop_last = True,
        ) 
        total_iter = len(trainset) / batch_size_list[-1]
        if generator.alpha < 1:
            generator.alpha_step = (1 - generator.alpha) / (training_epochs - start_epoch) / (2*total_iter)
            discriminator.alpha_step = (1 - discriminator.alpha) / (training_epochs - start_epoch) / (2*total_iter)

    # Losses
    iter_loss_G = 0.0
    iter_loss_D = 0.0
    iter_num = 0
    current_resolution = pow(2, generator.num_blocks + 1)
    for epoch in range(start_epoch, training_epochs):
        generator.train()
        if epoch in epoch_list:
            '''
            if current epoch number is included in epoch_list,
            the model's network should be grown up
            '''
            if pow(2, generator.num_blocks + 1 ) < resolution:
                '''
                list.index(value) : find the first index of value
                list.where(value) : find all the indices of value.
                '''
                num = epoch_list.index(epoch)
                # Change batch size.
                train_dataloader = torch.utils.data.DataLoader(
                    dataset = trainset,
                    batch_size = batch_size_list[num],
                    shuffle = True,
                    drop_last = True,
                )
                total_iter = len(trainset) / batch_size_list[num]
                generator.grow_model(growing_time_list[num]*total_iter)
                generator.grow_model(growing_time_list[num]*total_iter)
                current_resolution = pow(2, generator.num_blocks + 1)
                # print(f'Current output resuloution : {current_resolution} X {current_resolution}')
        
        print(f'Epoch : [{epoch}/{training_epochs}]')
        data_bar = tqdm(train_dataloader)
        for iter, samples in enumerate(data_bar):
            '''
            sample = (images, labels)
            '''
            ####
            # Train Disriminator
            ####
            optimizer_D.zero_grad()
            if current_resolution != resolution:
                samples = F.interpolate(samples[0], size=current_resolution).to(device)
            else:
                samples = samples[0].to(device)
            latent_z = torch.randn(samples.shape[0], latent_dim, 1, 1, device=device)
            fake_image = generator(latent_z)
            fake_pred = discriminator(fake_image.detach())
            real_pred = disriminator(samples)

            # Calculate gradient penalty.
            eps = torch.randn(samples.shape[0], 1, 1, 1, deivice=device)
            eps = eps.expand_as(samples)
            x_hat = eps*samples + (1-eps)*fake_image.detach()
            x_hat.requires_grad = True
            x_hat_pred = disriminator(x_hat)
            gradient = torch.autograd.grad(outputs=x_hat_pred.sum(), inputs=x_hat, create_graph=True)[0]
            gradient_norm = grad.view(samples.shape[0], -1).norm(2, dim=1)
            gradient_penalty = lambda_*(pow(2, (gradient_norm - 1))).mean()

            loss_D = fake_pred.mean() - real_pred.mean() + gradient_penalty
            loss_D.backward()
            optimizer_D.step()
            iter_loss_D += loss_D.item()

            ####
            # Train Generator
            ####
            optimizer_G.zero_grad()
            fake_pred = discriminator(fake_image)
            loss_G = -fake_pred.mean()
            loss_G.backward()
            optimizer_G.step()
            iter_loss_G += loss_G.item()

            iter_num += 1
            if iter % 500 == 0:
                iter_loss_G /= iter_num
                iter_loss_D /= iter_num
                print(f'Iteration : {iter}, GP : {gradient_penalty}')
                data_bar.set_description(f'loss_G: {iter_loss_G:>.3f}\tloss_D: {iter_loss_D:>.3f}')
                iter_num = 0
                iter_loss_G = 0.0
                iter_loss_D = 0.0
                
        save_checkpoint(
            model=generator, 
            name='G', 
            opt=optimizer_G, 
            epoch=epoch, 
            num_blocks=generator.num_blocks, 
            alpha=generator.alpha, 
            path=checkpoints_path)
        save_checkpoint(
            model=discriminator,
            name='D', 
            opt=optimizer_D, 
            epoch=epoch, 
            num_blocks=discriminator.num_blocks, 
            alpha=discriminator.alpha, 
            path=checkpoints_path)

        with torch.no_grad():
            generator.eval()
            fake_image = generator(fixed_latent_z)
            save_image(
                fake_image.data, 
                os.path.join(samples_path, f'fake_sample_{epoch+1}_{current_resolution}x{current_resolution}.png'), 
                nrow=4, 
                padding=0, 
                normalize=True)