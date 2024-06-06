"""
Source: https://github.com/rishikksh20/CrossViT-pytorch
"""
import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from bitnet.config import DataParams
from bitnet.models.crossvit_modules import (
    Attention,
    CrossAttention,
    FeedForward,
    PreNorm,
)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.attention_layer = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.feed_forward_layer = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        _input = self.attention_layer(_input) + _input
        _input = self.feed_forward_layer(_input) + _input
        return _input


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            _input = layer(_input)
        return _input


class MultiScaleTransformerEncoder(nn.Module):
    def __init__(self, small_dim: int, small_depth: int, small_heads: int, small_dim_head: int,
                 small_mlp_dim: int, large_dim: int, large_depth: int, large_heads: int, large_dim_head: int,
                 large_mlp_dim: int, cross_attn_depth: int, cross_attn_heads: int, dropout: float) -> None:
        super().__init__()
        self.transformer_enc_small = Transformer(small_dim, small_depth, small_heads, small_dim_head, small_mlp_dim)
        self.transformer_enc_large = Transformer(large_dim, large_depth, large_heads, large_dim_head, large_mlp_dim)

        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(
                    large_dim, heads=cross_attn_heads, dim_head=large_dim_head, dropout=dropout
                )),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(
                    small_dim, heads=cross_attn_heads, dim_head=small_dim_head, dropout=dropout
                )),
            ]) for _ in range(cross_attn_depth)
        ])


    def forward(self, input_small: torch.Tensor, input_large: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_small = self.transformer_enc_small(input_small)
        input_large = self.transformer_enc_large(input_large)

        for i in range(len(self.cross_attn_layers)):
            layers: nn.ModuleList = self.cross_attn_layers[i] #type: ignore

            small_to_large_1 = layers[0]
            large_to_small_1 = layers[1]
            cross_attn_small = layers[2]
            large_to_small_2 = layers[3]
            small_to_large_2 = layers[4]
            cross_attn_large = layers[5]

            small_class = input_small[:, 0]
            x_small = input_small[:, 1:]
            large_class = input_large[:, 0]
            x_large = input_large[:, 1:]

            # Cross Attention for Large Patch
            cal_q_large = large_to_small_2(large_class.unsqueeze(1))
            cal_qkv_large = torch.cat((cal_q_large, x_small), dim=1)
            cal_out_large = cal_q_large + cross_attn_large(cal_qkv_large)
            cal_out_large = small_to_large_2(cal_out_large)
            input_large = torch.cat((cal_out_large, x_large), dim=1)

            # Cross Attention for Small Patch
            cal_q_small = small_to_large_1(small_class.unsqueeze(1))
            cal_qkv_small = torch.cat((cal_q_small, x_large), dim=1)
            cal_out_small = cal_q_small + cross_attn_small(cal_qkv_small)
            cal_out_small = large_to_small_1(cal_out_small)
            input_small = torch.cat((cal_out_small, x_small), dim=1)

        return input_small, input_large


class CrossViT(nn.Module):
    def __init__(self, image_size: int, channels: int, num_classes: int,
                 patch_size_small: int = 14, patch_size_large: int = 16, small_dim:int = 96,
                 large_dim:int = 192, small_depth:int = 1, large_depth:int = 4, cross_attn_depth:int = 1,
                 multi_scale_enc_depth:int = 3, heads:int = 3,
                 pool = 'cls', dropout: float = 0., emb_dropout:float = 0., scale_dim:int = 4,
                 weights: None = None):
        super().__init__()
        if weights is not None:
            raise NotImplementedError(f"We don't have pretrained weights for {self.__class__.__name__}")
        assert image_size % patch_size_small == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_small = (image_size // patch_size_small) ** 2
        patch_dim_small = channels * patch_size_small ** 2

        assert image_size % patch_size_large == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_large = (image_size // patch_size_large) ** 2
        patch_dim_large = channels * patch_size_large ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding_small = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_small, p2 = patch_size_small),
            nn.Linear(patch_dim_small, small_dim),
        )

        self.to_patch_embedding_large = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size_large, p2=patch_size_large),
            nn.Linear(patch_dim_large, large_dim),
        )

        self.pos_embedding_small = nn.Parameter(torch.randn(1, num_patches_small + 1, small_dim))
        self.cls_token_small = nn.Parameter(torch.randn(1, 1, small_dim))
        self.dropout_small = nn.Dropout(emb_dropout)

        self.pos_embedding_large = nn.Parameter(torch.randn(1, num_patches_large + 1, large_dim))
        self.cls_token_large = nn.Parameter(torch.randn(1, 1, large_dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(
                MultiScaleTransformerEncoder(
                    small_dim=small_dim, small_depth=small_depth,
                    small_heads=heads, small_dim_head=small_dim//heads,
                    small_mlp_dim=small_dim*scale_dim,
                    large_dim=large_dim, large_depth=large_depth,
                    large_heads=heads, large_dim_head=large_dim//heads,
                    large_mlp_dim=large_dim*scale_dim,
                    cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                    dropout=dropout
                )
            )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(small_dim),
            nn.Linear(small_dim, num_classes)
        )

        self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(large_dim),
            nn.Linear(large_dim, num_classes)
        )


    def forward(self, _input:Tensor) -> Tensor:
        out_small: Tensor = self.to_patch_embedding_small(_input)
        batch_size, nchannel, _ = out_small.shape

        cls_token_small = repeat(self.cls_token_small, '() n d -> b n d', b = batch_size)
        out_small = torch.cat((cls_token_small, out_small), dim=1)
        out_small += self.pos_embedding_small[:, :(nchannel + 1)]
        out_small = self.dropout_small(out_small)

        out_large: Tensor = self.to_patch_embedding_large(_input)
        batch_size, nchannel, _ = out_large.shape

        cls_token_large = repeat(self.cls_token_large, '() n d -> b n d', b = batch_size)
        out_large = torch.cat((cls_token_large, out_large), dim=1)
        out_large += self.pos_embedding_large[:, :(nchannel + 1)]
        out_large = self.dropout_large(out_large)

        for multi_scale_transformer in self.multi_scale_transformers:
            out_small, out_large = multi_scale_transformer(out_small, out_large)

        out_small = out_small.mean(dim = 1) if self.pool == 'mean' else out_small[:, 0]
        out_large = out_large.mean(dim = 1) if self.pool == 'mean' else out_large[:, 0]

        out_small = self.mlp_head_small(out_small)
        out_large = self.mlp_head_large(out_large)
        out = out_small + out_large
        return out


def crossvit_base(**kwargs):
    model = CrossViT(image_size = DataParams.input_size[0], channels = DataParams.num_channels, **kwargs)
    return model