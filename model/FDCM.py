import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from timm.models.layers import to_2tuple
from PIL import Image


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x





class frefusion(nn.Module):
    def __init__(self, dim, size, reweight_expansion_ratio=.25, num_filters=8): # , bias=False, sparsity_threshold=0.01, **kwargs,
        super(frefusion, self).__init__()
        self.size = size
        self.filter_size = size//2 + 1
        self.dim = dim
        # self.conv_r = nn.Conv2d(dim,dim,1,1,0)
        # self.act_r = StarReLU()
        # self.conv_t = nn.Conv2d(dim,dim,1,1,0)
        # self.act_t = StarReLU()
        self.conv_r = nn.Sequential(nn.Conv2d(dim,dim,1,1,0),
                      nn.BatchNorm2d(dim),
                      StarReLU())
        self.conv_t = nn.Sequential(nn.Conv2d(dim,dim,1,1,0),
                      nn.BatchNorm2d(dim),
                      StarReLU())

        self.num_filters = num_filters
        # self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * dim)
        # self.complex_weights = nn.Parameter(torch.randn(self.size, self.filter_size, num_filters, 2, dtype=torch.float32) * 0.02)

        # self.sparsity_threshold = sparsity_threshold
        self.rc_aEnhance = nn.Sequential(
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))
        self.rc_pEnhance = nn.Sequential(
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))
        self.tc_aEnhance = nn.Sequential(
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))
        self.tc_pEnhance = nn.Sequential(
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2 + 1, dim // 2 + 1, 1, 1, 0))
        # self.act = nn.Identity
        self.DwconvFFN = nn.Sequential(nn.BatchNorm2d(dim),
                                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False))

        self.conv_phase_r = nn.Sequential(
            nn.Conv2d(dim,dim,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(dim,dim,1,1,0))
        self.conv_phase_t = nn.Sequential(
            nn.Conv2d(dim,dim,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(dim,dim,1,1,0))

        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(dim, dim//3, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim//3),
            nn.ReLU(inplace=True),
        )

        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(dim, dim//3, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim//3),
            nn.ReLU(inplace=True),
        )
    def forward(self, r, t):
        B, H, W, _ = r.shape   # 1,96,96,96
        x = r + t
        # routeing = self.reweight(x.mean(dim=(2, 3))).view(B, self.num_filters, -1).softmax(dim=1)  # 1,4,96

        res_r = r
        res_t = t
        r = self.conv_r(r)
        # r = self.act_r(r)
        t = self.conv_t(t)
        # t = self.act_t(t)
        r = r.to(torch.float32) # 1,64,56,56
        t = t.to(torch.float32)

        r_fft = torch.fft.rfft2(r, dim=(2, 3), norm='ortho') # 1,64,56,29
        t_fft = torch.fft.rfft2(t, dim=(2, 3), norm='ortho')
        rs_a = torch.abs(r_fft)
        rs_p = torch.angle(r_fft)
        ts_a = torch.abs(t_fft)
        ts_p = torch.angle(t_fft)

        rs_fft = torch.fft.rfft2(rs_a, dim=1, norm='ortho')
        ts_fft = torch.fft.rfft2(ts_a, dim=1, norm='ortho')
        rsc_a = torch.abs(rs_fft)
        rsc_p = torch.angle(rs_fft)
        tsc_a = torch.abs(ts_fft)
        tsc_p = torch.angle(ts_fft)
        rsc_aEnh = self.rc_aEnhance(rsc_a)
        rsc_pEnh = self.rc_pEnhance(rsc_p)
        tsc_aEnh = self.tc_aEnhance(tsc_a)
        tsc_pEnh = self.tc_pEnhance(tsc_p)
        rsc_r = rsc_aEnh * torch.cos(rsc_pEnh)
        rsc_i = rsc_aEnh * torch.sin(rsc_pEnh)
        rsc_comp = torch.complex(rsc_r, rsc_i)
        tsc_r = tsc_aEnh * torch.cos(tsc_pEnh)
        tsc_i = tsc_aEnh * torch.sin(tsc_pEnh)
        tsc_comp = torch.complex(tsc_r, tsc_i)
        rc = torch.fft.irfft2(rsc_comp, dim=1, norm='ortho')
        tc = torch.fft.irfft2(tsc_comp, dim=1, norm='ortho')

        rsc = rc * rs_a
        tsc = tc * ts_a
        rs_p = self.conv_phase_r(rs_p)
        rs_r = rsc * torch.cos(rs_p)
        rs_i = rsc * torch.sin(rs_p)
        rs_comp = torch.complex(rs_r, rs_i)
        ts_p = self.conv_phase_t(ts_p)
        ts_r = tsc * torch.cos(ts_p)
        ts_i = tsc * torch.sin(ts_p)
        ts_comp = torch.complex(ts_r, ts_i)
        rt_s = rs_comp + ts_comp       # 1,96,96,49                                  # c
        # # weight = torch.view_as_complex(self.complex_weights) # 96,96,49,4
        # weight = torch.view_as_complex(self.complex_weights) # 96,49,4
        # routeing = routeing.to(torch.complex64)              # 1,4,96
        # # weights = torch.einsum('bfc,chwf->bchw', routeing, weight)
        # weights = torch.einsum('bfc,hwf->bchw', routeing, weight)
        # weights = weights.view(-1, self.dim, self.size, self.filter_size)  # 1,56,29,128

        # rt_sEnh = rt_s * weights
        # rt_sEnh = rt_s * weight
        rt = torch.fft.irfft2(rt_s, dim=(2, 3), norm='ortho')
        out_1 = rt + res_r + res_t
        out = self.DwconvFFN(out_1) + out_1
        # out = self.DwconvFFN(rt) + rt

        # out_l = tensor2freq_image(out,pass_manner='low')
        # out_h = tensor2freq_image(out,pass_manner='high')
        #
        # out_l = self.outconv_bn_relu_L(out_l)
        # out_h = self.outconv_bn_relu_H(out_h)

        return out



def tensor2freq_image( x, pass_manner):
    passband = 20
    if pass_manner == 'high':
        freq = torch.fft.fft2(x, dim=[2,3], norm='ortho')
        freq = torch.fft.fftshift(freq)
        H, W = x.size(2), x.size(3)
        h_crop, w_crop = int(H / 2), int(W / 2)
        freq[:, :, h_crop - passband:h_crop + passband, w_crop - passband:w_crop + passband] = 0
    elif pass_manner == 'low':
        freq = torch.fft.fft2(x, dim=[2,3], norm='ortho')
        freq = torch.fft.fftshift(freq)
        H, W = x.size(2), x.size(3)
        h_crop, w_crop = int(H / 2), int(W / 2)
        mask = torch.zeros((x.size(0), x.size(1), H, W), dtype=torch.uint8).cuda()
        mask[:, :, h_crop - passband:h_crop + passband, w_crop - passband:w_crop + passband] = 1
        freq = freq * mask
    # else:
    #     freq = torch.fft.fft2(x, dim=[2, 3], norm='ortho')
    #     freq = torch.fft.fftshift(freq)
    fred = torch.abs(torch.fft.ifft2(freq, dim=(2, 3),norm='ortho'))

    return fred




if __name__ == '__main__':
    a = np.random.random((1,384,96,96))
    b = np.random.random((1,384,96,96))
    c = torch.Tensor(a).cuda()
    d = torch.Tensor(b).cuda()
# #     net = CAVER_R50D().cuda()
# #     print(net)
# #     out = net(c , d)
# #     print(out.shape)
    data = {'image':c,'depth':d}
    net = frefusion(384).cuda()
    out = net(c,d)
    print(out)

