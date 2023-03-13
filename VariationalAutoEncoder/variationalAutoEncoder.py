import os

import torch
import torch.nn as nn

from torchvision import transforms, datasets

from model import VariationalAutoEncoder

def save_checkpoint(model, opt, epoch, path):   
    torch.save({
        'model':model.state_dict(),     
        'optimizer': opt.state_dict(),  
        'epoch': epoch                  
    }, os.path.join(path, f'model_{epoch}.tar'))

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    
    learning_rate = 0.001
    training_epochs = 30
    batch_size = 128

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    training_batches = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = batch_size,
        shuffle = True,
    )

    model = VariationalAutoEncoder().to(device)
    reconstruction_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_batch = len(training_batches)

    # Training loop
    for epoch in range(training_epochs):
        avg_cost = 0.0
        for X, Y_label in training_batches:
            X = X.to(device)
            mu, log_var = model.encoder(X)
            z = model.reparameterize(mu, log_var)
            reconstructed = model.decoder(z)

            rec_loss = reconstruction_loss(reconstructed, X)
            KLD_loss = torch.mean(-0.5*torch.sum(1 + log_var-mu**2 - log_var.exp(), dim=1), dim=0)
            loss = rec_loss + KLD_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_cost += loss / total_batch
        save_checkpoint(model, optimizer, epoch, checkpoints_path)
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


