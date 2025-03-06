
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common.box_filter import BoxFilter
from basicseg.utils.registry import NET_REGISTRY
from basicseg.networks.resnet import resnet50
import math

# 深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# DepthwiseSeparableConv(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False if norm == nn.BatchNorm2d else True),

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        # self.conv_layer = DepthwiseSeparableConv(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
        #                                   out_channels=out_channels * 2, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            # self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x.to(torch.float32), dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0].to(torch.float32), ffted[..., 1].to(torch.float32))

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

class FourierUnit1(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit1, self).__init__()
        self.groups = groups
        self.conv_layer = DepthwiseSeparableConv(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            # self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)
        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        if self.use_se:
            ffted = self.se(ffted)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)
        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            #DepthwiseSeparableConv(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)
        #self.conv2 = DepthwiseSeparableConv(out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class SpectralTransform1(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform1, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit1(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit1(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = DepthwiseSeparableConv(out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        # print(x_l.shape)
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg

class FFC1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC1, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else DepthwiseSeparableConv
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else DepthwiseSeparableConv
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else DepthwiseSeparableConv
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform1
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else DepthwiseSeparableConv
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        # print(x_l.shape)
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            # print((self.convl2g(x_l)).shape)
            # print(l2g_gate)
            # print((self.convg2g(x_g)).shape)
            #x_g = x_g[:, :, :32, :32]  # 裁剪x_g使其与self.convl2g的输出形状相匹配
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)[:, :, :32, :32]

        return out_xl, out_xg

class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, padding_mode='zeros',dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=False, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class FFC_BN_ACT1(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=False, **kwargs):
        super(FFC_BN_ACT1, self).__init__()
        self.ffc = FFC1(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 inline=False, outline=False,**conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.inline = inline
        self.outline= outline
    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.outline:
            out = torch.cat(out, dim=1)
        return out

class FFCResnetBlock1(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 inline=False, outline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT1(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT1(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.inline = inline
        self.outline= outline
    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.outline:
            out = torch.cat(out, dim=1)
        return out

#初始
class CDC_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, stride=1,
                padding=1, dilation=1, theta=0.7, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                            stride = stride, dilation=dilation, bias=bias, padding_mode=padding_mode)
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        if (self.theta - 0.0) < 1e-6:
            return norm_out
        else:
            # [c_out, c_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            diff_out = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                dilation=1, padding=0)
            out = norm_out - self.theta * diff_out
            return out


# 增加对中心点所在的行和列分别进行卷积的功能
class CDC_conv1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, stride=1,
                padding=1, dilation=1, theta=0.7, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                            stride=stride, dilation=dilation, bias=bias, padding_mode=padding_mode)
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        if (self.theta - 0.0) < 1e-6:
            return norm_out
        else:
            # 原始
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            diff_out = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                dilation=1, padding=0)
            # 计算行方向的卷积
            kernel_diff_row = self.conv.weight.sum(2)
            kernel_diff_row = kernel_diff_row[:, :, :, None]
            diff_out_row = F.conv2d(input=x, weight=kernel_diff_row, bias=self.conv.bias, stride=self.conv.stride,
                                    dilation=1, padding=(self.conv.padding[0], 0))

            # 计算列方向的卷积
            kernel_diff_col = self.conv.weight.sum(3)
            kernel_diff_col = kernel_diff_col[:, :, None, :]
            diff_out_col = F.conv2d(input=x, weight=kernel_diff_col, bias=self.conv.bias, stride=self.conv.stride,
                                    dilation=1, padding=(0, self.conv.padding[1]))

            # 将行和列方向的结果结合
            diff_out = diff_out + diff_out_row + diff_out_col

            # 计算最终输出
            out = norm_out - self.theta * diff_out
            return out

# 水平卷积
class HorizontalConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, stride=1, padding=1, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding),
                              stride=stride, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)
# 垂直卷积
class VerticalConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, stride=1, padding=1, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0),
                              stride=stride, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)

class CDC_conv2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, stride=1,
                 padding=1, dilation=1, theta=0.7, padding_mode='zeros'):
        super().__init__()
        self.original_cdc = CDC_conv(in_channels, out_channels // 2, bias=bias, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, theta=theta,
                                            padding_mode=padding_mode)
        self.horizontal_conv = HorizontalConv(in_channels, out_channels // 4, bias=bias, kernel_size=kernel_size,
                                              stride=stride, padding=padding, padding_mode=padding_mode)
        self.vertical_conv = VerticalConv(in_channels, out_channels // 4, bias=bias, kernel_size=kernel_size,
                                          stride=stride, padding=padding, padding_mode=padding_mode)
        #self.adaptive_weight = AdaptiveWeight(out_channels, out_channels)

    def forward(self, x):
        original_out = self.original_cdc(x)
        horizontal_out = self.horizontal_conv(x)
        vertical_out = self.vertical_conv(x)
        out1 = torch.cat((horizontal_out, vertical_out), dim=1)
        out = torch.cat((original_out, out1), dim=1)
        return out

class Conv2dHoriVeriCross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2dHoriVeriCross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.zeros(C_out, C_in, 1, device=x.device)
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros,
                                 self.conv.weight[:, :, :, 1], self.conv.weight[:, :, :, 2],
                                 self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros), dim=2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if abs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff

class Conv2dDiagCross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2dDiagCross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff
# CombinedConv
class CDC_conv3(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, stride=1,
                 padding=1, dilation=1, theta=0.7, padding_mode='zeros'):
        super().__init__()
        self.cdc_conv = CDC_conv(in_channels, out_channels // 2, bias=bias, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, theta=theta,
                                padding_mode=padding_mode)
        self.hvc_conv = Conv2dHoriVeriCross(in_channels, out_channels // 4, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            bias=bias, theta=theta)
        self.dc_conv = Conv2dDiagCross(in_channels, out_channels // 4, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            bias=bias, theta=theta)

    def forward(self, x):
        cdc_out = self.cdc_conv(x)
        hvc_out = self.hvc_conv(x)
        dc_out = self.dc_conv(x)
        out1 = torch.cat((hvc_out, dc_out), dim=1)
        out = torch.cat((cdc_out, out1), dim=1)
        #out = torch.cat((cdc_out, hvc_out), dim=1)
        #out = torch.cat((cdc_out, dc_out), dim=1)
        return out


#标准卷积
class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='zeros'):
        super(StandardConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)


# 分组卷积
class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups, kernel_size=3, stride=1, padding=1, bias=True,
                 padding_mode='zeros'):
        super(GroupConv, self).__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)

#   # 使用示例
#
# groups = min(in_c // 2, out_c // 2)  # 假设我们希望至少将输入和输出通道分成两半
# conv_layer = GroupConv(in_c, out_c, groups=groups, kernel_size=3, padding=1, stride=stride, bias=False if norm == nn.BatchNorm2d else True)


# 版本2   滤波模块处理前后通道数不变
class ConvGuidedFilter(nn.Module):
    def __init__(self, in_channels, out_channels, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()
        self.box_filter = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=1)
        self.conv_a = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                    norm(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                    norm(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False))
        #nn.init.kaiming_normal_(self.box_filter.weight, mode='fan_out', nonlinearity='relu')
        self.box_filter.weight.data[...] = 1.0
    def forward(self, x_lr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, channels, _, _ = x_lr.size()  # 获取输入通道数
        N = self.box_filter(x_lr.data.new().resize_((1, channels, h_lrx, w_lrx)).fill_(1.0))
        mean_x = self.box_filter(x_lr) / N
        cov_xy = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x
        test = torch.cat([cov_xy, cov_xy], dim=1)
        A = self.conv_a(test)
        b = mean_x - A * mean_x
        mean_A = F.interpolate(A, (h_lrx, w_lrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_lrx, w_lrx), mode='bilinear', align_corners=True)
        return mean_A * x_lr + mean_b

#添加了一个可训练的权重向量 self.weights，同时修改了输出的计算方式。对于每个尺度，
# 我们分别计算出对应的 A 和 b，然后将它们插值到原始输入大小，并乘以对应的权重值。最后，将不同尺度的输出加权平均得到最终结果
class MultiScaleConvGuidedFilter3(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4], norm=nn.BatchNorm2d):
        super(MultiScaleConvGuidedFilter3, self).__init__()
        self.scales = scales
        #self.conv2d = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
        # self.conv2d = nn.Sequential(
        #         nn.Conv2d(out_channels * 3, out_channels * 2, kernel_size=3, stride=1, padding=1),
        #         norm(out_channels * 2),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
        #         norm(out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #     )
        #self.conv_fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # self.conv_fusion = nn.Sequential(
        #         nn.Conv2d(out_channels * 3, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=True),
        #         norm(out_channels * 2),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #         norm(out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        #     )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_a_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
            ) for _ in scales
        ])
        self.box_filters = nn.ModuleList([])
        for i in range(len(scales)):
            box_filter = nn.Conv2d(out_channels, out_channels, kernel_size=2 * scales[i] + 1, padding=scales[i],
                                   dilation=1,
                                   bias=False, groups=1)
            # nn.init.kaiming_normal_(box_filter.weight, mode='fan_out', nonlinearity='relu')
            box_filter.weight.data[...] = 1.0
            self.box_filters.append(box_filter)
        # Define the weight matrix for combining outputs from different scales
        self.weights = nn.Parameter(torch.ones(len(scales)), requires_grad=True)

    def forward(self, x_lr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, channels, _, _ = x_lr.size()
        # Prepare N (normalization factor)
        N_list = []
        for box_filter in self.box_filters:
            N = box_filter(x_lr.data.new().resize_((1, channels, h_lrx, w_lrx)).fill_(1.0))
            N_list.append(N)
        mean_x_list = []
        cov_xy_list = []
        for i, box_filter in enumerate(self.box_filters):
            mean_x = box_filter(x_lr) / N_list[i]
            cov_xy = box_filter(x_lr * x_lr) / N_list[i] - mean_x * mean_x
            mean_x_list.append(mean_x)
            cov_xy_list.append(cov_xy)
        # Feature fusion
        fused_features = []
        for cov_xy in cov_xy_list:
            fused_features.append(torch.cat([cov_xy, cov_xy], dim=1))
        # Learn the coefficients A and b
        A_list = []
        b_list = []
        for i, conv_a in enumerate(self.conv_a_list):
            A = conv_a(fused_features[i])
            b = mean_x_list[i] - A * mean_x_list[i]
            A_list.append(A)
            b_list.append(b)
        # Interpolate A and b to the original size and apply weights
        weighted_output_list = []
        for i, (A, b) in enumerate(zip(A_list, b_list)):
            A_resized = F.interpolate(A, (h_lrx, w_lrx), mode='bilinear', align_corners=True)
            b_resized = F.interpolate(b, (h_lrx, w_lrx), mode='bilinear', align_corners=True)
            weighted_output = self.weights[i] * (A_resized * x_lr + b_resized)
            weighted_output_list.append(weighted_output)
        # Combine the results from different scales
        #combined_output = sum(weighted_output_list) / len(self.scales)  # 1
        # 特征连接
        combined_output = torch.cat(weighted_output_list, dim=1)
        #combined_output = self.sigmoid(self.conv_fusion(combined_output))  # 2
        #combined_output = self.relu(self.bn(self.conv_fusion(combined_output)))  # 3
        combined_output = self.relu(self.conv_fusion(combined_output))  # 3
        #combined_output = self.sigmoid(self.conv2d(combined_output))  # 4
        #combined_output = self.relu(self.conv2d(combined_output))  # 5
        return combined_output

# 通道减半再拼接
class MultiScaleConvGuidedFilter4(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2], norm=nn.BatchNorm2d):
        super(MultiScaleConvGuidedFilter4, self).__init__()
        self.scales = scales
        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)  #  7/3
        self.conv_fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fusion1 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv_a_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
            ) for _ in scales
        ])
        self.box_filters = nn.ModuleList([])
        for i in range(len(scales)):
            box_filter = nn.Conv2d(out_channels, out_channels, kernel_size=2 * scales[i] + 1, padding=scales[i],
                                   dilation=1,
                                   bias=False, groups=1)
            # nn.init.kaiming_normal_(box_filter.weight, mode='fan_out', nonlinearity='relu')
            box_filter.weight.data[...] = 1.0
            self.box_filters.append(box_filter)
        # Define the weight matrix for combining outputs from different scales
        self.weights = nn.Parameter(torch.ones(len(scales)), requires_grad=True)

    def forward(self, x_lr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, channels, _, _ = x_lr.size()
        # Prepare N (normalization factor)
        N_list = []
        for box_filter in self.box_filters:
            N = box_filter(x_lr.data.new().resize_((1, channels, h_lrx, w_lrx)).fill_(1.0))
            N_list.append(N)
        mean_x_list = []
        cov_xy_list = []
        for i, box_filter in enumerate(self.box_filters):
            mean_x = box_filter(x_lr) / N_list[i]
            cov_xy = box_filter(x_lr * x_lr) / N_list[i] - mean_x * mean_x
            mean_x_list.append(mean_x)
            cov_xy_list.append(cov_xy)
        # Feature fusion
        fused_features = []
        for cov_xy in cov_xy_list:
            fused_features.append(torch.cat([cov_xy, cov_xy], dim=1))
        # Learn the coefficients A and b
        A_list = []
        b_list = []
        for i, conv_a in enumerate(self.conv_a_list):
            A = conv_a(fused_features[i])
            b = mean_x_list[i] - A * mean_x_list[i]
            A_list.append(A)
            b_list.append(b)
        # Interpolate A and b to the original size and apply weights
        weighted_output_list = []
        for i, (A, b) in enumerate(zip(A_list, b_list)):
            A_resized = F.interpolate(A, (h_lrx, w_lrx), mode='bilinear', align_corners=True)
            b_resized = F.interpolate(b, (h_lrx, w_lrx), mode='bilinear', align_corners=True)
            weighted_output = self.weights[i] * (A_resized * x_lr + b_resized)
            weighted_output_list.append(self.conv_fusion1(weighted_output))
        # Combine the results from different scales
        #combined_output = sum(weighted_output_list) / len(self.scales)  
        # 特征连接
        combined_output = torch.cat(weighted_output_list, dim=1)  # 1
        #out = torch.cat([weighted_output_list[0], weighted_output_list[1]], dim=1)  self.conv_fusion
        #combined_output = self.sigmoid(self.conv_fusion(combined_output)) # 2
        combined_output = self.relu(self.conv_fusion(combined_output)) # 3
        #combined_output = self.sigmoid(self.conv2d(combined_output))  # 4
        #combined_output = self.relu(self.conv2d(combined_output))  # 5

        return combined_output


# 无参注意力
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
        #return self.sigmoid(avgout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out




class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, theta_1=0, theta_2=0.7, theta_r=0., norm=nn.BatchNorm2d,
                 padding_mode='zeros'):
        super().__init__()
        self.conv_block = nn.Sequential(
            CDC_conv(in_c, out_c, kernel_size=3, padding=1, theta=theta_1, padding_mode=padding_mode,
            stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
            nn.ReLU(inplace=True),
            CDC_conv(out_c, out_c, kernel_size=3, padding=1, theta=theta_2, padding_mode=padding_mode,
            stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.residual_block = nn.Sequential(
            CDC_conv(in_c, out_c, kernel_size=3, padding=1, theta=theta_r, padding_mode=padding_mode,
            stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.relu = nn.ReLU(inplace=True)
        self.simam = simam_module()  # 实例化 simam_module

    def forward(self, x):
        #x = self.simam(x)
        conv_out = self.conv_block(x)
        # 添加
        #x = self.simam(x)  # 使用实例化后的 simam_module
        residual_out = self.residual_block(x)
        out = self.relu(conv_out + residual_out)
        return out

class ResidualBlock1(nn.Module):
    def __init__(self, in_c, out_c, stride=1, theta_1=0, theta_2=0.7, theta_r=0., norm=nn.BatchNorm2d,
                 padding_mode='zeros'):
        super().__init__()
        self.conv_block = nn.Sequential(
            DepthwiseSeparableConv(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_c, out_c, kernel_size=3, padding=1, stride=stride,
                                   bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.residual_block = nn.Sequential(
            DepthwiseSeparableConv(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.relu = nn.ReLU(inplace=True)
        self.simam = simam_module()  # 实例化 simam_module

    def forward(self, x):
        #x = self.simam(x)
        conv_out = self.conv_block(x)
        # 添加
        #x = self.simam(x)  # 使用实例化后的 simam_module
        residual_out = self.residual_block(x)
        out = self.relu(conv_out + residual_out)
        return out

class ResidualBlock2(nn.Module):
    def __init__(self, in_c, out_c, stride=1, theta_1=0, theta_2=0.7, theta_r=0., norm=nn.BatchNorm2d,
                 padding_mode='zeros'):
        super().__init__()
        self.conv_block = nn.Sequential(
            #DepthwiseSeparableConv(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            StandardConv(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
            nn.ReLU(inplace=True),
            StandardConv(out_c, out_c, kernel_size=3, padding=1, stride=stride,
                         bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.residual_block = nn.Sequential(
            #DepthwiseSeparableConv(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            StandardConv(in_c, out_c, kernel_size=3, padding=1, stride=stride,
                         bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.relu = nn.ReLU(inplace=True)
        self.simam = simam_module()  # 实例化 simam_module

    def forward(self, x):
        #x = self.simam(x)
        conv_out = self.conv_block(x)
        # 添加
        #x = self.simam(x)  # 使用实例化后的 simam_module
        residual_out = self.residual_block(x)
        out = self.relu(conv_out + residual_out)
        return out


# 初始
class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.squeeze = nn.Conv2d(in_c, out_c, kernel_size=1)

        self.up_block = nn.Sequential(
            nn.Conv2d(out_c*2, out_c, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )
    def forward(self, x, lateral):
        size = lateral.shape[-2:]
        x = self.squeeze(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        out = self.up_block(torch.cat((x, lateral), dim=1))
        return out

# new1
class UpsampleBlock1(nn.Module):
    def __init__(self, in_c, out_c, guided_filter_radius=1, scales=[1, 2], norm=nn.BatchNorm2d):  #scales=[1, 2, 4],scales=[1, 2]
        super().__init__()
        self.squeeze = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.conv_up = nn.ConvTranspose2d(out_c, out_c, kernel_size=4, stride=2, padding=1)
        self.cov_block = nn.Sequential(
            nn.Conv2d(out_c , out_c, stride=1, kernel_size=3, padding=1),
            norm(out_c),
            nn.ReLU(),
        )
        self.guided_filter = ConvGuidedFilter(in_c, out_c, radius=guided_filter_radius, norm=norm)
        self.mutl_guided_filter3 = MultiScaleConvGuidedFilter3(in_c, out_c, scales=scales, norm=norm)
        self.mutl_guided_filter4 = MultiScaleConvGuidedFilter4(in_c, out_c, scales=scales, norm=norm)
        # self.CA = ChannelAttentionModule(channel, ratio=16)
        self.CA1 = ChannelAttentionModule(in_c, ratio=16)
        self.CA2 = ChannelAttentionModule(out_c, ratio=16)
        self.up_block = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, stride=1, kernel_size=3, padding=1),
            norm(out_c),
            nn.ReLU(),
        )

    def forward(self, x, lateral):
        #print("x0",x.shape)
        # print(lateral.shape)
        #x = self.CA1(x) * x
        size = lateral.shape[-2:]
        x = self.squeeze(x)
        # 初始
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        #x = self.conv_up(x)
        #x = self.CA(x)*x
        #lateral = self.CA2(lateral)*lateral
        #print("x1",x.shape)
        # 使用引导滤波器处理 x
        #print("lateral", lateral.size())
        # lateral1 = self.guided_filter(lateral)
        # lateral2 = self.cov_block(lateral1)
        # lateral2 = self.guided_filter(lateral2)
        # lateral = lateral2 + lateral1

        #lateral = self.guided_filter(lateral)
        lateral = self.mutl_guided_filter3(lateral)
        #lateral = self.CA2(lateral) * lateral
        #x = self.guided_filterup(x, lateral)
        #print(x.shape)

        out = self.up_block(torch.cat((x, lateral), dim=1))
        #out = self.CA2(out)*out
        return out

class UpsampleBlock2(nn.Module):
    def __init__(self, in_c, out_c, guided_filter_radius=1, scales=[1, 2],norm=nn.BatchNorm2d):  #scales=[1, 2, 4],scales=[1, 2]
        super().__init__()
        #self.squeeze = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.squeeze = DepthwiseSeparableConv(in_c, out_c, kernel_size=1, bias=True)
        self.conv_up = nn.ConvTranspose2d(out_c, out_c, kernel_size=4, stride=2, padding=1)
        self.cov_block = nn.Sequential(
            nn.Conv2d(out_c , out_c, stride=1, kernel_size=3, padding=1),
            norm(out_c),
            nn.ReLU(),
        )
        self.guided_filter = ConvGuidedFilter(in_c, out_c, radius=guided_filter_radius, norm=norm)
        self.mutl_guided_filter3 = MultiScaleConvGuidedFilter3(in_c, out_c, scales=scales, norm=norm)
        self.mutl_guided_filter4 = MultiScaleConvGuidedFilter4(in_c, out_c, scales=scales, norm=norm)
        # self.CA = ChannelAttentionModule(channel, ratio=16)
        self.CA1 = ChannelAttentionModule(in_c, ratio=16)
        self.CA2 = ChannelAttentionModule(out_c, ratio=16)
        self.up_block = nn.Sequential(
            DepthwiseSeparableConv(out_c * 2, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_c),
            nn.ReLU(),
        )

    def forward(self, x, lateral):
        #x = self.CA1(x) * x
        size = lateral.shape[-2:]
        x = self.squeeze(x)
        # 初始
        #x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.conv_up(x) 
        #x = self.CA(x)*x
        #lateral = self.CA2(lateral)*lateral
        #print("x1",x.shape)
        # 使用引导滤波器处理 x
        #print("lateral", lateral.size())
        # lateral1 = self.guided_filter(lateral)
        # lateral2 = self.cov_block(lateral1)
        # lateral2 = self.guided_filter(lateral2)
        # lateral = lateral2 + lateral1

        #lateral = self.guided_filter(lateral)
        lateral = self.mutl_guided_filter3(lateral)
        #lateral = self.CA2(lateral) * lateral
        #x = self.guided_filterup(x, lateral)

        out = self.up_block(torch.cat((x, lateral), dim=1))
        #out = self.CA2(out)*out
        return out

class ResidualFusionModule(nn.Module):
    def __init__(self, in_channels_low, in_channels_high):
        super(ResidualFusionModule, self).__init__()
        self.conv_low = nn.Conv2d(in_channels_low, in_channels_high, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels_high)
        self.conv_up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=4, stride=2, padding=1)

        self.f1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.f2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.conv_fusion0 = nn.Conv2d(in_channels_high, in_channels_high, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fusion1 = nn.Conv2d(in_channels_high, in_channels_high, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv_fusionc0 = nn.Conv2d(in_channels_high * 2, in_channels_high, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fusionc1 = nn.Conv2d(in_channels_high * 2, in_channels_high, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x_low, x_high):
        # 对低分辨率特征进行卷积，使其通道数与高分辨率特征相同
        x_low = self.conv_low(x_low)
        # 转置卷积上采样
        x_low = self.conv_up(x_low)
        # print("Shape of self.f1:", self.f1.shape)
        # print("Shape of self.f2:", self.f2.shape)
        # print("Shape of x_low:", x_low.shape)
        # print("Shape of x_high:", x_high.shape)
        # # 双线性插值上采样低分辨率特征
        # size = x_high.shape[-2:]
        # x_low = F.interpolate(x_low, size=size, mode='bilinear', align_corners=False)
        #x_c = torch.cat((self.f1 * x_low, self.f2 * x_high), dim=1)
        # 将上采样后的低分辨率特征与高分辨率特征相加
        #x = self.relu(self.f1 * x_low + self.f2 * x_high) # 1
        x = self.relu(self.bn(self.conv_fusion0(self.f1 * x_low + self.f2 * x_high))) # 2 best628
        #x = self.relu(self.bn(self.conv_fusion0(self.f1 * x_low + x_high)))
        #x = self.relu(self.conv_fusion0(self.f1 * x_low + self.f2 * x_high)) # 2 
        #x = self.relu(self.conv_fusion1(self.f1 * x_low + self.f2 * x_high)) # 3
        #x = self.f1 * x_low + self.f2 * x_high # 4
        #x =  x_low + x_high # 5
        #x = self.relu(self.conv_fusionc0(x_c)) # 6
        #x = self.relu(self.conv_fusionc1(x_c))  # 7
        return x

class FusionModule(nn.Module):  # 对同级特征融合
    def __init__(self, in_channels_low):
        super(FusionModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.f1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.f2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.conv_fusion0 = nn.Conv2d(in_channels_low, in_channels_low, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fusion1 = nn.Conv2d(in_channels_low, in_channels_low, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv_fusion2 = nn.Conv2d(in_channels_low * 2, in_channels_low, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fusion3 = CDC_conv3(in_channels_low * 2, in_channels_low, kernel_size=1, stride=1, padding=0, bias=True)
        self.ca = ChannelAttentionModule(in_channels_low)
        self.bn = nn.BatchNorm2d(in_channels_low)
        # self.conv_fusionc0 = nn.Conv2d(in_channels_high * 2, in_channels_high, kernel_size=1, stride=1, padding=0, bias=True)
        # self.conv_fusionc1 = nn.Conv2d(in_channels_high * 2, in_channels_high, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        #x = self.relu(self.f1 * x_low + self.f2 * x_high) # 1
        #x = self.relu(self.bn(self.conv_fusion0(self.f1 * x1 + self.f2 * x2))) # 2
        #x = self.relu(self.conv_fusion0(self.f1 * x1 + self.f2 * x2)) # 2 
        #x = self.relu(self.conv_fusion0(x1 + x2)) # 2 best628
        #x = self.relu(self.bn(self.conv_fusion0(x1 + x2)))

        x = self.relu(self.conv_fusion2(torch.cat((x1, x2), dim=1))) 
        #x = self.relu(self.conv_fusion3(torch.cat((x1, x2), dim=1)))
        return x

@NET_REGISTRY.register()
class DAGFNet(nn.Module):
    def __init__(self, in_c=3, out_c=1,
                base_dim=32,
                theta_0=0.7, theta_1=0, theta_2=0.7, theta_r=0,
                maxpool='pool', norm='bn', padding_mode='reflect',
                n_blocks=7, gt_ds = False,
                 ):
        super(UCFNet, self).__init__()
        # 在这里修改sigma1和sigma2的初始值
        self.gt_ds = gt_ds
        self.sigma1 = nn.Parameter(torch.tensor(1.0), requires_grad=True) # False 
        self.sigma2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # self.a1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # self.a2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d 
        if maxpool == 'pool':
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stride = 1
        self.conv1 = nn.Sequential(
            #ConvGuidedFilter(in_c, base_dim),
            # ConvGuidedFilter(),
            CDC_conv2(in_c, base_dim, bias=False, theta=theta_0),
            # ConvGuidedFilter(),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.conv10 = nn.Sequential(
            CDC_conv(16, 16, bias=False, theta=theta_0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            CDC_conv2(base_dim, base_dim, bias=False, theta=theta_0),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.conv20 = nn.Sequential(
            CDC_conv2(base_dim, base_dim, bias=False, theta=theta_0),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.conv22 = nn.Sequential(
            CDC_conv(base_dim, base_dim, bias=False, kernel_size=5, padding=2, theta=theta_0),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.conv23 = nn.Sequential(
            #CDC_conv(base_dim, base_dim, bias=False, kernel_size=5, padding=2, theta=theta_0),
            nn.Conv2d(base_dim, base_dim, stride=1, kernel_size=5, padding=2),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        #CDC_conv(base_dim, base_dim, bias=False, kernel_size=5, padding=2, theta=theta_0)
        self.conv11 = nn.Sequential(
            #DepthwiseSeparableConv(in_c, base_dim, kernel_size=3, padding=1, bias=False),
            StandardConv(in_c, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.conv21 = nn.Sequential(
            #DepthwiseSeparableConv(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            StandardConv(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResidualBlock(base_dim, base_dim * 2, norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=self.stride)
        self.layer2 = ResidualBlock(base_dim * 2, base_dim * 4, norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=self.stride)
        self.layer3 = ResidualBlock(base_dim * 4, base_dim * 8, norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=self.stride)
        self.layer4 = ResidualBlock(base_dim * 8, base_dim * 16, norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=self.stride)
        self.up3 = UpsampleBlock1(base_dim * 16, base_dim * 8)
        self.up2 = UpsampleBlock1(base_dim * 8, base_dim * 4)
        self.up1 = UpsampleBlock1(base_dim * 4, base_dim * 2)
        self.up0 = UpsampleBlock1(base_dim * 2, base_dim)
        #self.up00 = UpsampleBlock(base_dim , 16)
        #self.channel_attention = ChannelAttentionModule(base_dim)
        #self.fus1 = FusionModule(base_dim * 2)
        #self.fus2 = FusionModule(base_dim * 4)
        #self.fus3 = FusionModule(base_dim * 8)
        #self.fusion1 = ResidualFusionModule(base_dim * 2, base_dim)
        #self.fusion2 = ResidualFusionModule(base_dim * 4, base_dim * 2)
        #self.fusion3 = ResidualFusionModule(base_dim * 8, base_dim * 4)
        #self.spatial_attention = SpatialAttentionModule()
        #out = self.channel_attention(x) * x

        #self.last_conv1 = SpatialAttentionModule()
        self.last_conv = nn.Conv2d(base_dim, out_c, kernel_size=1, stride=1)
        #self.out1 = nn.Conv2d(base_dim * 2, out_c, kernel_size=1, stride=1)
        #self.out2 = nn.Conv2d(base_dim * 4, out_c, kernel_size=1, stride=1)
        #self.out3 = nn.Conv2d(base_dim * 8, out_c, kernel_size=1, stride=1)
        #self.out4 = nn.Conv2d(base_dim * 16, out_c, kernel_size=1, stride=1)
  
        ffc_blocks = nn.ModuleList()
        resnet_conv_kwargs = {'ratio_gin': 0.75, 'ratio_gout': 0.75}
        for i in range(n_blocks):
            if i == 0:
                cur_resblock = FFCResnetBlock(base_dim * 16, padding_type='reflect', activation_layer=nn.ReLU,
                                          norm_layer=nn.BatchNorm2d, inline=True, outline=False, **resnet_conv_kwargs)
            elif i == n_blocks-1:
                cur_resblock = FFCResnetBlock(base_dim * 16, padding_type='reflect', activation_layer=nn.ReLU,
                                          norm_layer=nn.BatchNorm2d, inline=False, outline=True, **resnet_conv_kwargs)
            else:
                cur_resblock = FFCResnetBlock(base_dim * 16, padding_type='reflect', activation_layer=nn.ReLU,
                                          norm_layer=nn.BatchNorm2d, inline=False, outline=False, **resnet_conv_kwargs)
            ffc_blocks.append(cur_resblock)
        self.ffc_blocks = nn.Sequential(*ffc_blocks)
        self.simam = simam_module()  # 实例化 simam_module
    def forward(self, x):
        out_00 = self.conv1(x)
        out_0 = self.conv2(out_00)
        out_0 = self.conv20(out_0)
        out_1 = self.layer1(self.maxpool(out_0))
        out_2 = self.layer2(self.maxpool(out_1))
        out_3 = self.layer3(self.maxpool(out_2))
        out_4 = self.layer4(self.maxpool(out_3))
        # print("out0", out_0.size())
        # print("out1", out_1.size())
        #fout_0 = self.fusion1(out_1, out_0)
        #fout_1 = self.fusion2(out_2, out_1)


        #fout_2 = self.fusion3(out_3, out_2)
        #ft2 = self.fus2(fout_2, out_2)
        #fout_1 = self.fusion2(ft2, out_1)
        #ft1 = self.fus1(fout_1, out_1)
        #fout_0 = self.fusion1(ft1, out_0)


        # fout_1 = self.fusion2(fout_2, out_1)
        # fout_0 = self.fusion1(fout_1, out_0)
        out_da = self.ffc_blocks(out_4)
        #out4 = F.interpolate(self.out4(out_da), scale_factor=16, mode='bilinear', align_corners=True)
        up_3 = self.up3(out_da, out_3)
        #out3 = F.interpolate(self.out3(up_3), scale_factor=8, mode='bilinear', align_corners=True)
        up_2 = self.up2(up_3, out_2)
        #out2 = F.interpolate(self.out2(up_2), scale_factor=4, mode='bilinear', align_corners=True)
        up_1 = self.up1(up_2, out_1)
        #out1 = F.interpolate(self.out1(up_1), scale_factor=2, mode='bilinear', align_corners=True)
        up_0 = self.up0(up_1, out_0)
        out = self.last_conv(up_0)
        #if self.gt_ds:
            #return out, out1, out2, out3, out4
        #else:
            #return out

        return out


def main():
    x = torch.rand(1, 3, 512, 512)
    net = UCFNet(theta_r=0, theta_0=0.7, theta_1=0, theta_2=0.7, n_blocks=7)
    y = net(x)
    import ptflops

    macs, params = ptflops.get_model_complexity_info(net, (3, 512, 512), print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # torch.onnx.export(net, x, 'uranet.onnx')


    # macs, params = ptflops.get_model_complexity_info(cdc_conv, (3, 512, 512))
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # model = nn.Sequential()
    # pred = model(x)
    # print(torch.equal(pred,x))
if __name__ == '__main__':
    main()
