import argparse
import os

import torch
import torch.nn as nn

import config

def get_loader(dataset, batch_size, resolution):
    transform = transforms.Compose([
        
    ])
def train(
        generator, 
        discriminator, 
        g_optim, 
        d_optim, 
        dataset, 
        step, 
        iteration, 
        start_point, 
        used_sample, 
        d_losses, 
        g_losses, 
        alph):
    # image resolution.
    resolution = 4 * 2 ** step
    

if __name__=='__main__':
    pass