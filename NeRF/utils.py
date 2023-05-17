import numpy as np
import torch

def get_encoder():
    pass

def hierarchical_volume_sampling(num_samples, weights, det=False, pytest=False):
    '''
    Hierarchical volume sampling
    num_samples : number of samples.
    weights     : weights.
    det         : what is it?
    pytest      : what is it?
    '''
    # Produce a piecewise-constant PDF along the ray.
    weights = weights + 1e-5    # prevent NaN.
    PDF = weights / torch.sum(weights, -1, keepdim=True)
    CDF = torch.cumsum(PDF, -1)
    # dimension 이해 안됨.
    CDF = torch.cat([torch.zeros_like(CDF[...,:1]), CDF], -1)   # (batch, len(bins))

    # Take uniform samples.
    if det:
        samples = torch.linspace(0., 1., steps=num_samples)
        samples = samples.expand(list(CDF.shape[:-1] + [num_samples]))
    else:
        samples = torch.rand(list(CDF.shape[:-1] + [num_samples]))

    # What is Pytest?
    if pytest:
        np.random.seed(0)
        shape = list(CDF.shape[:-1]) + [num_samples]
        if det:
            samples = np.linspace(0., 1., num_samples)
            samples = np.broadcast_to(samples, shape)
        else:
            samples = np.random.rand(*shape)
        samples = torch.Tensor(samples)
    
    # Inverse CDF
    samples = samples.contiguous()
    indices = torch.searchsorted(CDF, samples, right=True)
    below = torch.max(torch.zeros_like(indices-1), indices-1)
    above = torch.min((CDF.shape[-1]-1) * torch.ones_like(indices), indices)
    # What is indices_g?
    indices_g = torch.stack([below, above], -1) # (batch, num_samples, 2)