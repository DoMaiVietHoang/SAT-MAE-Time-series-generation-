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
    def __init__(self):
        # Initialize learneable positional embeddings
        # For sequence length 5: [CLS, pos1, pos2, pos3 ,pos4]

        self.postion_embedding = nn.Parameter(torch.randn(1,5,256))
    
    def create_timestamp_embeddings(self, timestamps):
        """
        Practice: Create time stamp embeddings from 3D timestamp (each dimmension cresspond the frequency image)
        Args: 
            timestamps: shape(B,N,3) 
        Returns:
            ts_embedding : timestamp embeddings ready for concat with pos_embed
        """
        flattened = timestamps.reshape(-1,3)
        embedding_0 = get_1d_sincos_pos_embed(128, flattened[:,0])
        embedding_1 = get_1d_sincos_pos_embed(128, flattened[:,1])
        embedding_2 = get_1d_sincos_pos_embed(128, flattened[:,2])
        timeseries_embedding = torch.stack([embedding_0,embedding_1,embedding_2], dim=1)  #
        print(timeseries_embedding.shape)
        B, N        = timestamps.shape[0], timestamps.shape[1]
        print(B)
        print(N)
        timeseries_embedding = timeseries_embedding.reshape(B,N,3,128)
        ts_embed = timeseries_embedding.reshape(B,N*3,-1)
        cls_embed= torch.zeros(B,1,128)
        ts_embed = torch.cat([cls_embed, ts_embed], dim=1)
        print(f"After adding CLS token: {ts_embed.shape}")
        return ts_embed

#####Test

B, N = 2, 4  # 2 batches, 4 tokens
timestamps = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],  # batch 0
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]  # batch 1
    ], dtype=torch.float32)
practice = Embedding()
ts_embed = practice.create_timestamp_embeddings(timestamps)
    
print(f"\nFinal timestamp embeddings shape: {ts_embed.shape}")
print("Expected: (2, 13, 128)")  # 2 batches, 13 tokens (1 CLS + 12), 128 d