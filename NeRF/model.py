import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, in_channels, out_channels, in_views, depth, width, skips, use_viewdirs):
        super(NeRF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_views = in_views
        self.depth = depth
        self.width = width
        self.skips = skips
        self.use_viewdirs = use_viewdirs