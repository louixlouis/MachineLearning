import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class NeRFModule(nn.Module):
    def __init__(self, in_pos_dim:int, in_view_dim:int, depth:int, w_dim:int, skips=[4]) -> None:
        super(NeRFModule, self).__init__()
        self.in_pos_dim = in_pos_dim
        self.in_view_dim = in_view_dim
        self.depth = depth
        self.w_dim = w_dim
        self.skips = skips

        # Define layers.
        self.position_layers = nn.ModuleList([nn.Linear(in_pos_dim, w_dim)] + [nn.Linear(w_dim, w_dim) if i not in self.skips else nn.Linear(w_dim + in_pos_dim, w_dim) for i in range(self.depth - 1)])
        self.view_dir_layers = nn.Linear(in_pos_dim + w_dim, w_dim//2)
        self.feature_layers = nn.Linear(w_dim, w_dim)
        self.density_layers = nn.Linear(w_dim, 1)
        self.color_layers = nn.Linear(w_dim//2, 3)
    
    def forward(self, x):
        in_position, in_direction = torch.split(x, [self.in_pos_dim, self.in_view_dim], dim=-1)
        out = in_position
        for i, layer in enumerate(self.position_layers):
            out = self.position_layers[i](out)
            out = F.relu(out)
            if i in self.skips:
                out = torch.cat([in_position, out], dim=-1)

        density = self.density_layers(out)
        feature = self.feature_layers(out)

        out = self.view_dir_layers(torch.cat([feature, in_direction], dim=-1))
        out = F.relu(out)
        color = self.color_layers(out)
        out = torch.cat([color, density], dim=-1)
        return out
    
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
        self.model_coarse = None
        self.model_fine = None
    
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x, is_fine:bool=False):
        if is_fine:
            return x
        else:
            return x