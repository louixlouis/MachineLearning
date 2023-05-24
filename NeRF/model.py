import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class NeRF(nn.Module):
    def __init__(self, in_pos_dim:int, in_view_dim:int, depth:int, w_dim:int, skips=[4], gt_camera_param=None) -> None:
        '''
        in_pos_dim : dimension of position (x, y, z)
        in_view_dim : dimension of camera view direction d
        depth : number of layers
        w_dim : dimension of latent w, default=256
        skips : location of skip connection.
        gt_camera_param :  
        '''
        super(NeRF, self).__init__()
