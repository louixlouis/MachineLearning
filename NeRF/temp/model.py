import torch
import torch.nn as nn
import torch.nn.functional as F

############
# Positional encoding
############
class PositionEncoder():
    def __init__(self, in_dim, max_freq, num_freqs, log_sampling):
        self.in_dim = in_dim
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling # Need ?
        
        self.encoder()

    def encoder(self):
        gamma_function = []
        freq_list = torch.linspace(2.**0., 2.**self.max_freq, steps=self.num_freqs)
        for freq in freq_list:
            for function in [torch.sin, torch.cos]:
                gamma_function.append(lambda x, function=function, freq=freq : function(freq*x))
        self.gamma_function = gamma_function

    def encoding(self, x):
        return torch.cat([function(x) for function in self.gamma_function], -1)

############
# NeRF model
############
class NeRF(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, in_views=3, depth=8, feature_dim=256, skips=[4], use_viewdirs=False):
        '''
        skips :
        use_viewdirs : 

        '''
        super(NeRF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_views = in_views
        self.depth = depth
        self.feature_dim = feature_dim
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # Linear layers for position.
        position_layers = [nn.Linear(self.in_channels, self.feature_dim)]
        for num in range(self.depth-1): # Why -1?
            if num not in skips:
                position_layers.append(nn.Linear(self.feature_dim, self.feature_dim))
            else:
                position_layers.append(nn.Linear(self.feature_dim + self.in_channels, self.feature_dim))
        self.position_layers = nn.ModuleList(position_layers)
        
        # Linear layers for view.
        # self.view_layers = nn.ModuleList([nn.Linear(in_views + feature_dim, feature_dim//2)])
        self.view_layer = nn.Linear(in_views + feature_dim, feature_dim//2)
        if use_viewdirs:
            self.feature_linear = nn.Linear(feature_dim, feature_dim)
            self.sigma_linear = nn.Linear(feature_dim, 1)
            self.rgb_linear = nn.Linear(feature_dim//2, 3)
        else:
            self.output_linear = nn.Linear(feature_dim, self.out_channels)
        
    def forward(self, x):
        '''
        torch.split(tensor, size or section, dim=0)
        size or section = size of a single chunk or list of sizes for each chunk.
        dim = dimension along which to split the tensor.
        '''
        in_positions, in_views = torch.split(x, [self.in_channels, self.in_views], dim=-1)
        out = in_positions
        for i, layer in enumerate(self.position_layers):
            out = layer(out)
            out = F.relu(out)
            if i in self.skips:
                # 논문 설명이랑 다르지 않나..?
                out = torch.cat([in_positions, out], -1)
        
        if self.use_viewdirs:
            # sigma is the volume density.
            # 1 dim
            sigma = self.sigma_linear(out)

            # Camera ray's viewing direction concatenated with feature vector.
            feature = self.feature_linear(out)
            out = torch.cat([feature, in_views], -1)
            
            out = self.view_layer(out)
            out = F.relu(out)

            # c is RGB vector
            # 3 dim
            c = self.rgb_linear(out)
            
            # 4 dim
            output = torch.cat([c, sigma], -1)
        else:
            output = self.output_linear(out)
        return output

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_view_dirs = False"
    