from typing import Callable

from einops import rearrange
from torch import Tensor, einsum, nn


class PreNorm(nn.Module):
    def __init__(self, dim: int, module: Callable):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module


    def forward(self, _input: Tensor, **kwargs) -> Tensor:
        return self.module(self.norm(_input), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout:float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


    def forward(self, _input: Tensor) -> Tensor:
        return self.net(_input)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_query_key_value = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, _input: Tensor) -> Tensor:
        _, _, _, num_heads = *_input.shape, self.heads
        qkv: Tensor = self.to_query_key_value(_input).chunk(3, dim = -1)
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = num_heads), qkv)

        dots: Tensor = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out: Tensor =  self.to_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads:int, dim_head: int, dropout:float):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale: float = float(dim_head) ** -0.5

        self.to_key = nn.Linear(dim, inner_dim , bias=False)
        self.to_value = nn.Linear(dim, inner_dim , bias = False)
        self.to_query = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x_qkv: Tensor) -> Tensor:
        _, _, _, num_heads = *x_qkv.shape, self.heads

        key: Tensor = self.to_key(x_qkv)
        key = rearrange(key, 'b n (h d) -> b h n d', h = num_heads)

        value: Tensor = self.to_value(x_qkv)
        value = rearrange(value, 'b n (h d) -> b h n d', h = num_heads)

        query: Tensor = self.to_query(x_qkv[:, 0].unsqueeze(1))
        query = rearrange(query, 'b n (h d) -> b h n d', h = num_heads)

        dots = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out: Tensor =  self.to_out(out)
        return out
