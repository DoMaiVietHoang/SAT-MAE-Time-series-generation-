import torch
import torch.nn as nn
import math
import numpy as np

def get_1d_sincos_pos_embed(embed_dim:int, pos:torch.tensor):
    """
    Create 1D posional Embedding 
    Args:
        embed_dim: embedding dimmension of time posititon
        pos      : positons
    
    """
    assert embed_dim%2==0
    omega = torch.arange(embed_dim//2,dtype = torch.float32)
    omega /= embed_dim/2.
    omega = 1. / 1000 * omega
    pos = pos.reshape(-1)
    out = torch.einsum('n,d->nd', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    embedding = torch.cat([emb_sin,emb_cos],dim=1)
    return embedding

class Embedding:
    
