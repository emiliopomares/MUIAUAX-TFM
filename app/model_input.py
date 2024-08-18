import torch

IMG_SIZE = 256
N_CHANNELS = 6
BATCH_SIZE = 32 # Let's stick to the classics

def make_model_input(l, r, permute=True):
    """Creates a tensor input datapoint to be fed into the model
    from l and r images
    Parameters:
    - l: left image
    - r: right image
    - permute (bool): apply permutation
    Returns:
    torch.tensor
    """
    if permute:
        assert l.shape == (IMG_SIZE, IMG_SIZE, N_CHANNELS//2)
        assert r.shape == (IMG_SIZE, IMG_SIZE, N_CHANNELS//2)
        lr = torch.cat([l, r], dim=2).permute(2,0,1).float()
    else:
        assert l.shape == (N_CHANNELS//2, IMG_SIZE, IMG_SIZE)
        assert r.shape == (N_CHANNELS//2, IMG_SIZE, IMG_SIZE)
        lr = torch.cat([l, r], dim=0).float()
    return lr