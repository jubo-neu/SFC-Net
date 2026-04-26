
import torch
import torch.nn as nn
from einops import rearrange
# from thop import profile
# from thop import clever_format
import numpy as np



class CBR(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
    ):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=nn.ReLU(inplace=False))

class CBRBlock(nn.Sequential):
    def __init__(self, in_c, out_c, num_blocks=1, kernel_size=3):
        assert num_blocks >= 1
        super().__init__()

        if kernel_size == 3:
            kernel_setting = dict(kernel_size=3, stride=1, padding=1)
        elif kernel_size == 1:
            kernel_setting = dict(kernel_size=1)
        else:
            raise NotImplementedError

        cs = [in_c] + [out_c] * num_blocks
        self.channel_pairs = self.slide_win_select(cs, win_size=2, win_stride=1, drop_last=True)
        self.kernel_setting = kernel_setting

        for i, (i_c, o_c) in enumerate(self.channel_pairs):
            self.add_module(name=f"cbr_{i}", module=CBR(i_c, o_c, **self.kernel_setting))

    @staticmethod
    def slide_win_select(items, win_size=1, win_stride=1, drop_last=False):
        num_items = len(items)
        i = 0
        while i + win_size <= num_items:
            yield items[i : i + win_size]
            i += win_stride

        if not drop_last:
            # 对于������������不满一个win_size的切片，保留
            yield items[i : i + win_size]


class FFN(nn.Module):
    def __init__(self, dim, out_dim=None, mul=1):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            CBRBlock(dim, dim * mul, num_blocks=2, kernel_size=3),
            nn.Conv2d(dim * mul, out_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

class spafusion(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None): # num_heads=8
        super().__init__()
        self.bn_dim = nn.BatchNorm2d(dim)
        self.bn_2dim = nn.BatchNorm2d(dim*2)
        self.PConv = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 # **: mi yun suan
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1) # 8 1
        self.lnx = nn.LayerNorm(dim)
        self.lny = nn.LayerNorm(dim)
        self.lnz = nn.LayerNorm(dim)
        self.bn = nn.BatchNorm2d(1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.ffn = FFN(dim=2 * dim, mul=1, out_dim=2 * dim)
    def forward(self, x, y):
        batch_size = x.shape[0] # 4
        chanel = x.shape[1] # 64
        xy = torch.cat((x, y), dim=1)
        xy = self.PConv(xy)
        x = self.bn_dim(x)
        y = self.bn_dim(y)
        xy = self.bn_dim(xy)
        sc = x # input: r2_ (4, 64, 24, 24)
        sd = y

        x = x.view(batch_size, chanel, -1).permute(0, 2, 1) # (4, 576, 64)
        sc1 = x # (4, 576, 64)
        x = self.lnx(x)

        y = y.view(batch_size, chanel, -1).permute(0, 2, 1) # (4, 576, 64)
        sd1 = y
        y = self.lny(y) # (4, 576, 64)


        xy = xy.view(batch_size, chanel, -1).permute(0, 2, 1)
        xy = self.lnz(xy)


        B, N, C = x.shape # B4, N576, C64
        x_qv = self.qv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y_qv = self.qv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_v = x_qv[0], x_qv[1]
        y_q, y_v = y_qv[0], y_qv[1]

        xyk = self.k(xy).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = xyk[0]



        attn1 = (x_q @ k.transpose(-2, -1)) * self.scale  # @ is matrix multiplication, y_k.transpose(-2, -1): (4, 8, 8, 576), attn(4, 8, 576, 576)
        attn2 = (y_q @ k.transpose(-2, -1)) * self.scale

        attn = attn1 * attn2#   attn1 * attn2
        attn = attn.softmax(dim=-1)

        x = (attn @ x_v).transpose(1, 2).reshape(B, N, C) # attn @ x_v: (4, 8, 576, 8)   transpose(1, 2): (4, 576, 8, 8)  reshape: (4,576, 64)
        x = self.proj(x) # (4,576, 64)
        y = (attn @ y_v).transpose(1,2).reshape(B, N, C)
        y = self.proj(y)

        x = (x + sc1) # (4,576, 64)
        x = x.permute(0, 2, 1) # (4, 64, 574)
        x = x.view(batch_size, chanel, *sc.size()[2:]) # (4, 64, 24, 24)
        x = self.conv2(x) + x # (4, 64, 24, 24)

        y = (y + sd1) # (4,576, 64)
        y = y.permute(0, 2, 1) # (4, 64, 574)
        y = y.view(batch_size, chanel, *sd.size()[2:]) # (4, 64, 24, 24)
        y = self.conv2(y) + y # (4, 64, 24, 24)

        add_Gfea = x + y
        mul_Gfea = x * y
        cat_Gfea = torch.cat((add_Gfea, mul_Gfea), dim=1)
        cat_Gfea = cat_Gfea + self.ffn(self.bn_2dim(cat_Gfea))


        # return x, y, self.act(self.bn(self.conv(attn + attn.transpose(-1, -2))))
        return cat_Gfea