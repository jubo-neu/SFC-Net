import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

from model.CSAM import csam
from model.MFFM import CDFF
from model.SBIM import spafusion
from model.FDCM import frefusion
from torchvision.models.feature_extraction import create_feature_extractor

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class FFM(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(FFM, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()
class Enhance(nn.Module):
    def __init__(self,in_c):
        super(Enhance, self).__init__()
        self.conv0 = nn.Conv2d(in_c,in_c, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(in_c)

    def forward(self, input1, input2):
        out = F.relu(self.bn0(self.conv0(input1 + input2)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Split(nn.Module):
    def __init__(self,c_in=256, c_out=128):
        super(Split, self).__init__()
        self.br2 = nn.Sequential(
            BasicConv(c_in, c_out, kernel_size=1, bias=False, bn=True, relu=True),

            BasicConv(c_out, c_out, kernel_size=3, dilation=1, padding=1, groups=c_out, bias=False,
                      relu=False),
        )

        self.conv1b = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(c_out)

        self.conv1d = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn1d = nn.BatchNorm2d(c_out)


    def forward(self, x):

        out2 = self.br2(x)
        out1b = F.relu(self.bn1b(self.conv1b(out2)), inplace=True)
        out1d = F.relu(self.bn1d(self.conv1d(out2)), inplace=True)

        return out1b, out1d


class sodnet(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 decode_channels1=32,
                 dropout=0.1,
                 backbone_name="unireplknet_b_in22k_to_in1k_384_acc86.44",
                 # backbone_name="resnet101d",
                 # backbone_name="swin_base_patch4_window12_384_in22k",
                 # backbone_name="vgg16",
                 # backbone_name="convnext_tiny.in12k_ft_in1k_384,convnextv2_base.fcmae_ft_in22k_in1k_384",
                 pretrained=True,
                 window_size=8,

                 ):
        super().__init__()
        backbone_full = timm.create_model(model_name=backbone_name, pretrained=False)
        try:
            state_dict = torch.load(r"/home/cjb123321/projects/SFCNet/pretrained/unireplknet_b_in22k_to_in1k_384_acc86.44.pth", map_location='cpu')
            backbone_full.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}. Using random initialization.") 
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv_sem3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv_sem2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)    
        self.conv4sem = BasicConv(2048, 1024, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        return_nodes = {
            'stages.0': 'res1',
            'stages.1': 'res2',
            'stages.2': 'res3',
            'stages.3': 'res4'
        }
        self.backbone = create_feature_extractor(backbone_full, return_nodes=return_nodes)
        self.fusion = VSSBlock_fuse(hidden_dim=1024)
        self.curvelet_prior = CurveletBoundaryPrior()
        self.mid_reduce = nn.Sequential(
            nn.Conv2d(3 * decode_channels, decode_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True)
        )
        self.lambda_geom = 0.5
        # self.conv2 = ConvBN(256, decode_channels, kernel_size=1)
        # self.conv3 = ConvBN(512, decode_channels, kernel_size=1)
        # self.conv4 = ConvBN(1024, decode_channels, kernel_size=1)

        self.conv1 = BasicConv(256, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv2 = BasicConv(256, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv3 = BasicConv(512, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv4 = BasicConv(1024, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)

        # self.conv1_ = BasicConv(128, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
        #               relu=False)
        self.conv2_ = BasicConv(256, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv3_ = BasicConv(512, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv4_ = BasicConv(1024, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)

        self.feature_sf = CDFF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        # self.feature_dt = CDFF(in_dim=decode_channels, out_dim=decode_channels//2, num_head=2,w=12)

        # self.SF2GL = SFM(in_ch=decode_channels*2, out_ch=decode_channels, num_heads=8, window_size=window_size)
        self.csam = csam(decode_channels)
        self.ffm1 = FFM(in_channels=decode_channels, decode_channels=decode_channels)
        self.ffm2 = FFM(in_channels=decode_channels, decode_channels=decode_channels)

        self.split = Split(c_in=decode_channels*2, c_out=decode_channels)
        self.enhance_b = Enhance(in_c=decode_channels)
        self.enhance_d = Enhance(in_c=decode_channels)

        self.down1 = BasicConv(3 * decode_channels, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels, bias=False,
                      relu=False)
        self.down2 = BasicConv(3 * decode_channels, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels, bias=False,
                      relu=False)
        self.down3 = BasicConv(6 * decode_channels, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels, bias=False,
                      relu=False)
        self.spafuse = spafusion(3 * decode_channels, num_heads=8)
        self.spafuse_ = spafusion(1024, num_heads=8)
        self.frefuse = frefusion(3 * decode_channels, 48)
        self.linearb1 = nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1)
        self.lineard1 = nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1)
        self.linearb2 = nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1)
        self.lineard2= nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1)
        self.linear1 = nn.Sequential(nn.Conv2d(decode_channels*2, decode_channels, kernel_size=3, padding=1), nn.BatchNorm2d(decode_channels),
                                    nn.ReLU(inplace=True), nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1))
        self.linear2 = nn.Sequential(nn.Conv2d(decode_channels * 2, decode_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(decode_channels),
                                    nn.ReLU(inplace=True), nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1))

    def forward(self, x, y, imagename=None):

        features_x = self.backbone(x)
        features_y = self.backbone(y)

        res1, res2, res3, res4 = features_x['res1'], features_x['res2'], features_x['res3'], features_x['res4']
        tes1, tes2, tes3, tes4 = features_y['res1'], features_y['res2'], features_y['res3'], features_y['res4']
        
        res1h, res1w = res1.size()[-2:]  # 96,96
        res2h, res2w = res2.size()[-2:]

        semantic = self.fusion(res4, tes4)

        res4 = res4 * semantic
        tes4 = tes4 * semantic
        res3 = res3 * self.conv_sem3(self.up2(semantic))
        tes3 = tes3 * self.conv_sem3(self.up2(semantic))
        res2 = res2 * self.conv_sem2(self.up4(semantic))
        tes2 = tes2 * self.conv_sem2(self.up4(semantic))
        
        # res1 = self.conv1(res1)
        # tes1 = self.conv1_(tes1)
        res2 = self.conv2(res2)
        tes2 = self.conv2_(tes2)
        res3 = self.conv3(res3)
        tes3 = self.conv3_(tes3)
        res4 = self.conv4(res4)
        tes4 = self.conv4_(tes4)

        # res2 = F.interpolate(res2, size=(res2h, res2w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res2h, res2w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res2h, res2w), mode='bicubic', align_corners=False)
        mid_res = torch.cat([res2, res3, res4], dim=1)

        # tes2 = F.interpolate(tes2, size=(res2h, res2w), mode='bicubic', align_corners=False)
        tes3 = F.interpolate(tes3, size=(res2h, res2w), mode='bicubic', align_corners=False)
        tes4 = F.interpolate(tes4, size=(res2h, res2w), mode='bicubic', align_corners=False)
        mid_tes = torch.cat([tes2, tes3, tes4], dim=1)#1,384,96,96

        rt1_add = res1 + tes1
        rt1_mul = res1 * tes1
        rt1 = torch.cat([rt1_add, rt1_mul], dim=1)
        rt1 = self.conv1(rt1)

        spa_fuse = self.spafuse(mid_res,mid_tes)#1,256,96,96
        fre_fuse = self.frefuse(mid_res, mid_tes)#1,128,96,96
        spa_fuse = self.down3(spa_fuse)
        fre_fuse = self.down2(fre_fuse)
        bd = self.feature_sf(spa_fuse, fre_fuse)
        glb, local = self.csam(bd)#1,128,48,48

        mid_res = self.down1(mid_res)#1,128,96,96
        mid_tes = self.down1(mid_tes)

        res = mid_res + glb
        res = F.interpolate(res, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res = self.ffm1(res, rt1)#1,128,96,96

        tes = mid_tes + local
        tes = F.interpolate(tes, size=(res1h, res1w), mode='bicubic', align_corners=False)
        tes = self.ffm2(tes, rt1)#1,128,96,96

        out1 = torch.cat([res, tes], dim=1)#1,2,96,96
        outb2, outd2 = self.split(out1)
        outb2 = self.enhance_b(res, outb2)
        outd2 = self.enhance_d(tes, outd2)
        out2 = torch.cat([outb2, outd2], dim=1)

        shape = x.size()[2:]
        out1 = F.interpolate(self.linear1(out1), size=shape, mode='bilinear')
        outb1 = F.interpolate(self.linearb1(res), size=shape, mode='bilinear')
        outd1 = F.interpolate(self.lineard1(tes), size=shape, mode='bilinear')

        out2 = F.interpolate(self.linear2(out2), size=shape, mode='bilinear')
        outb2 = F.interpolate(self.linearb2(outb2), size=shape, mode='bilinear')
        outd2 = F.interpolate(self.lineard2(outd2), size=shape, mode='bilinear')

        return outb1, outd1, out1, outb2, outd2, out2

if __name__ == '__main__':
    a = np.random.random((1, 3, 384, 384))
    b = np.random.random((1, 3, 384, 384))
    c = torch.Tensor(a).cuda()
    d = torch.Tensor(b).cuda()
    data = {'image': c, 'depth': d}
    net = sodnet().cuda()
    out = net(c, d)
    # flops, params = profile(net, inputs=(c, d))
    # flops, params = clever_format([flops, params], '%.3f')
    # print(flops, params)
