import os

import numpy as np

import torch

from load_blender import *
from model import *

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # Hyper parameters.
    learning_rate = 0.0002
    epochs = 20
    batch_size = 64
    data_type = 'blender'
    data_root = ''
    downsample = None
    test_skip = None
    bg_white = None
    L_x = 10
    L_d = 4
    depth = 8
    w_dim = 256

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    # Data load.
    print('Blender data')
    if data_type == 'blender':
        images, gt_camera_params, HW, i_split = load_blender(
            root=data_root, 
            downsample=downsample, 
            test_skip=test_skip, 
            bg_white=bg_white)
    elif data_type == 'llff':
        pass
    elif data_type == 'custom':
        pass

    i_train, i_val, i_test = i_split
    height, width = HW
    gt_intrinsic, gt_extrinsic = gt_camera_params

    # Define model.
    # Positional encoding.
    pos_encode, in_channel = PositionEncoder(L_x)
    pos_encode_d, in_channel_d = PositionEncoder(L_d)

    model = NeRF(
        in_channel=in_channel, 
        in_channel_d=in_channel_d,
        depth=depth, 
        w_dim=w_dim).to(device)

    # Loss functions.
    loss = torch.nn.MSELoss()

    # Optimizers.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Scheduler.
    
if __name__ == '__main__':
    main()