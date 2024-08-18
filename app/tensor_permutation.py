import torch
import torch.nn.functional as F
import numpy as np

def get_padded_size(data, N):
    """
    Calculates padding sizes for F.pad so that each dimension is a multiple of N
    """
    n_dim = len(data.shape)
    dims = []
    for dim in range(n_dim):
        l = data.shape[dim]
        needed = ((N-(l-(l//N)*N))%N)
        needed_low = needed//2
        needed_high = needed-needed_low
        dims.append(needed_low)
        dims.append(needed_high)
    return tuple(dims[::-1])

def permute_target_tensor(t):
    return torch.flip(t, dims=[2]).permute(2, 0, 1)

def unpermute_target_tensor(t):
    return torch.flip(t.permute(1, 2, 0), dims=[2])

def crop_output(t):
    return t[6:6+37, 12:12+25, 15:15+18]

def expand_output(t):
    return F.pad(t, get_padded_size(np.zeros(t.shape), 48))