import copy

import torch
from torch import nn
import torch.nn.functional as F
import numbers
import einops
from einops import rearrange
from Model.SGFFN import SGFeedForward

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class channel_shuffle(nn.Module):
    def __init__(self):
        super(channel_shuffle, self).__init__()

    def forward(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups,
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac
        res = torch.softmax(res, dim=-1)
        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res


### Axis-based Multi-head Self-Attention (row and col attention)


class NextAttentionZ(nn.Module):

    def __init__(self, num_dims, num_heads=1, num_groups=4, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_groups = num_groups
        self.dims = int(num_dims/num_groups)
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.size()
        assert len(x.size()) == 4
        x1 = self.row_att(x)
        x1 = x1.transpose(-2, -1)
        x1 = self.col_att(x1)
        x1 = x1.transpose(-2, -1)
        out = x1.view(n, -1, h, w)
        return out








######  Axis-based Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, num_groups=4,  ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.num_groups = num_groups
        self.num_channel = int(dim/num_groups)
        self.norm1 = LayerNorm(self.num_channel, LayerNorm_type)
        self.attn = NextAttentionZ(self.num_channel, num_heads, num_groups=num_groups)
        self.norm2 = LayerNorm(self.num_channel, LayerNorm_type)
        self.ffn = SGFeedForward(self.num_channel, ffn_expansion_factor, bias)



    def forward(self, x):
        n, c, h, w = x.size()

        y = x.view(n*self.num_groups, self.num_channel, h, w)
        y = y + self.attn(self.norm1(y))
        y = y + self.ffn(self.norm2(y))
        y = y.view(n, -1, h, w)
        return y

class Transformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1, num_groups=4, shuffle_factor=0.5, ffn_expansion_factor=2.66):
        super().__init__()
        self.shuffle = channel_shuffle()
        self.shuffle_num = int(num_groups*shuffle_factor)
        self.layer = nn.ModuleList()
        for i in range(num_layers):
            layer = TransformerBlock(dim=dim,  num_heads=num_heads, num_groups=num_groups,
                                     ffn_expansion_factor=ffn_expansion_factor, bias=True, LayerNorm_type='WithBias')
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        for layer in self.layer:
            x = self.shuffle(x, self.shuffle_num)
            x = layer(x)

        return x


# shuffle = channel_shuffle()
# shuffle_num = int(2)
# img = torch.tensor[[1, 1, 1],
#                     [2,2,2]]
# print(img.shape)