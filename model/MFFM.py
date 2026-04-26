import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class CDFF(nn.Module):#Cross-domain feature fusion
    def __init__(self, dim, num_heads, LayerNorm_type, ):
        super(CDFF, self).__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(2*dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out3 = nn.Conv2d(2*dim, dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_1_3 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv1_2_3 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv2_1_1 = nn.Conv2d(2*dim, 2*dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_1_2 = nn.Conv2d(2*dim, 2*dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_1_3 = nn.Conv2d(2*dim, 2*dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2_1 = nn.Conv2d(2*dim, 2*dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv2_2_2 = nn.Conv2d(2*dim, 2*dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_2_3 = nn.Conv2d(2*dim, 2*dim, (21, 1), padding=(10, 0), groups=dim)



    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.norm3(x3)
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)
        attn_113 = self.conv1_1_3(x1)
        attn_121 = self.conv1_2_1(x1)
        attn_122 = self.conv1_2_2(x1)
        attn_123 = self.conv1_2_3(x1)

        attn_211 = self.conv1_1_1(x2)
        attn_212 = self.conv1_1_2(x2)
        attn_213 = self.conv1_1_3(x2)
        attn_221 = self.conv1_2_1(x2)
        attn_222 = self.conv1_2_2(x2)
        attn_223 = self.conv1_2_3(x2)

        attn_311 = self.conv2_1_1(x3)
        attn_312 = self.conv2_1_2(x3)
        attn_313 = self.conv2_1_3(x3)
        attn_321 = self.conv2_2_1(x3)
        attn_322 = self.conv2_2_2(x3)
        attn_323 = self.conv2_2_3(x3)


        out1 = attn_111 + attn_112 + attn_113 +attn_121 + attn_122 + attn_123
        out2 = attn_211 + attn_212 + attn_213 +attn_221 + attn_222 + attn_223
        out3 = attn_311 + attn_312 + attn_313 + attn_321 + attn_322 + attn_323
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        out3 = self.project_out3(out3)
        k3 = rearrange(out3, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v3 = rearrange(out3, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        # k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        # v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k3 = torch.nn.functional.normalize(k3, dim=-1)

        attn1 = (q1 @ k3.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out4 = (attn1 @ v3) + q1

        attn2 = (q2 @ k3.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out5 = (attn2 @ v3) + q2
        out4 = rearrange(out4, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out5 = rearrange(out5, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out4) + self.project_out(out5) + x1+x2

        return out


