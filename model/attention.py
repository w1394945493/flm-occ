
import torch
from torch import nn
import torch.nn.functional as F
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

from model import scene



# Move head forward and fold into batch dim. dimensions become (B, nh, S, hs)
def _split_heads(t: torch.Tensor, B: int, S: int, H: int, Hs: int):
    return t.view(B, S, H, Hs)


class MHA(nn.Module):

    def __init__(self, dim_mae, num_heads, custom_freqs, attn_type='self', dim_space=2, pos_emb_dim=21, dropout_p=0.0) -> None:
        super(MHA, self).__init__()
        self.dim_mae = dim_mae    # 128
        self.num_heads = num_heads    # 3
        self.head_dim = self.dim_mae // self.num_heads
        self.attn_type = attn_type
        self.dropout_p = dropout_p
        if attn_type == 'self':
            self.qkv_proj = nn.Linear(self.dim_mae, 3 * self.dim_mae, bias=True)
        else:
            self.q_proj = nn.Linear(self.dim_mae, self.dim_mae, bias=True)
            self.kv_proj = nn.Linear(self.dim_mae, 2 * self.dim_mae, bias=True)
        # self.resid_drop = nn.Dropout(config.resid_dropout, inplace=False)
        self.proj = nn.Linear(self.dim_mae, self.dim_mae, bias=True)
        
        self.dim_space = dim_space
        self.rotary_embeddings = RotaryEmbedding(pos_emb_dim, custom_freqs=custom_freqs)

    def rotary_positional_encode(self, x, voxel_index):
        """
        x: (B, N, Hs, D)
        voxel_index: (B, N, 3)
        """
        freqs_x = self.rotary_embeddings(voxel_index[:, :, 0])
        freqs_y = self.rotary_embeddings(voxel_index[:, :, 1])
        if self.dim_space == 2:
            freqs = torch.cat((freqs_x, freqs_y), dim=-1)
        elif self.dim_space == 3:
            freqs_z = self.rotary_embeddings(voxel_index[:, :, 2])
            freqs = torch.cat((freqs_x, freqs_y, freqs_z), dim=-1)
        else:
            raise ValueError(f"Unsupported dim_space: {self.dim_space}")
        freqs = freqs.unsqueeze(2).repeat(1, 1, self.num_heads, 1)
        return apply_rotary_emb(freqs, x, seq_dim=-3)

    def forward(self, x, y=None, x_index=None, y_index=None, attn_bias=None):

        B, S_q, _ = x.size()

        if self.attn_type =='self':
            qkv = self.qkv_proj(x)
            qkv = _split_heads(qkv, B, S_q, self.num_heads, 3 * self.head_dim)
            q, k, v = qkv.chunk(3, dim=-1)
            y_index = x_index
        else:
            S_kv = y.shape[1]
            q = self.q_proj(x)
            kv = self.kv_proj(y)
            q = _split_heads(q, B, S_q, self.num_heads, self.head_dim)
            kv = _split_heads(kv, B, S_kv, self.num_heads, 2 * self.head_dim)
            k, v = kv.chunk(2, dim=-1)
        
        q = self.rotary_positional_encode(q, x_index)
        k = self.rotary_positional_encode(k, y_index)
        # y = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        q, k, v = [ t.permute(0,2,1,3).contiguous() for t in (q, k, v) ]  # BLHD->BHLD
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.dropout_p)
        y = y.permute(0,2,1,3).contiguous()  # BHLD->BLHD

        y = y.to(q.dtype)
        y = y.flatten(start_dim=2, end_dim=3)
        # y = self.resid_drop(self.proj(y))
        y = self.proj(y)
        return y
    
class MHA3D(nn.Module):
    """
    Fake mha layer that actually uses a MHA block
    """
    def __init__(
        self, 
        in_channels,
        num_heads=4,
        pe_dim=21,
        custom_freqs=None,
        dropout_p=0.15,
    ):
        super().__init__()
        
        self.mha = MHA(in_channels, num_heads, custom_freqs, 'self', 3, pe_dim, dropout_p)
        
    def forward(self, BGSfeat, BGS, BK, BWH):
        """
        EGO's version
        """
        BXYZ = scene.Superquadrics.contracted_to_euclidean(BGS[..., :3], BK, BWH)
        indices = BXYZ

        x = BGSfeat
        x = self.mha(x, x_index=indices, attn_bias=None)
        
        return x

