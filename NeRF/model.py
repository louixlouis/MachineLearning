import torch
import torch.nn as nn

############
# Positional encoding
############
class PositionEncoder():
    def __init__(self, in_dim, max_freq, num_freqs, log_sampling):
        self.in_dim = in_dim
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling # Need ?
        
    def encoder(self):
        gamma_function = []
        freq_list = torch.linspace(2.**0, 2.**self.max_freq, steps=self.num_freqs)
        for freq in freq_list:
            for function in [torch.sin, torch.cos]:
                gamma_function.append(lambda x, function=function, freq=freq : function(freq*x))
        self.gamma_function = gamma_function

    

    
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
        
        # Linear layers for position.
        self.position_layer = []
        
        # Linear layers for view.
        self.view_layer = nn.Linear(in_views + width, width//2)
        
    def forward(self, x):
        '''
        torch.split(tensor, size or section, dim=0)
        size or section = size of a single chunk or list of sizes for each chunk.
        dim = dimension along which to split the tensor.
        '''
        in_position, in_view = torch.split(x, [self.in_channels, self.in_views], dim=-1)
    
    def temp():
        pass