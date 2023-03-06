# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnext import convnext_base
from torch.nn import Conv2d, BatchNorm2d


def conv3x3(in_planes, out_planes, groups=1, stride=1, padding=1, has_bias=False):
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=has_bias)


def conv1x1(in_planes, out_planes, groups=1):
    return Conv2d(in_planes, out_planes, kernel_size=1, groups=groups)


def conv3x3_bn_relu(in_planes, out_planes, groups=1, stride=1, padding=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, groups=groups, stride=stride, padding=padding),
            BatchNorm2d(out_planes),
            nn.ReLU(),
            )


def conv1x1_bn_relu(in_planes, out_planes, groups=1):
    return nn.Sequential(
            conv1x1(in_planes, out_planes, groups=groups),
            BatchNorm2d(out_planes),
            nn.ReLU(),
            )


class M1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(M1, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv2 = conv3x3_bn_relu(in_channel, out_channel)

    def forward(self, top):
        out = self.conv2(self.up(top))
        return out


class M(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(M, self).__init__()
        self.conv1 = conv3x3_bn_relu(in_channel, in_channel)
        self.up = nn.Upsample(scale_factor=2)
        self.conv2 = conv3x3_bn_relu(in_channel, out_channel)

    def forward(self, r, top):
        out = self.conv2(self.up(self.conv1(top) + r))
        return out


# feature extraction module
class FEM(nn.Module):
    def __init__(self, channel, dim, div=8):
        super(FEM, self).__init__()
        self.channel = channel
        self.dim = dim
        self.div = div
        self.convr = conv1x1_bn_relu(channel, dim)
        self.convd = conv1x1_bn_relu(channel, dim)
        self.conv1 = conv3x3_bn_relu(dim * 2, dim)
        self.conv_final = conv1x1_bn_relu(dim, channel)

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, r, d):
        r = self.convr(r)
        d = self.convd(d)
        r = F.interpolate(r, scale_factor=(1/self.div))
        d = F.interpolate(d, scale_factor=(1/self.div))

        add = r + d
        mul = r * self.sigmoid(d)
        cat = self.conv1(torch.cat([add, mul], dim=1))

        b, c, h, w = cat.size()
        cat_k = cat.view(b, self.dim, -1)
        cat_q = cat.view(b, self.dim, -1).transpose(-1, -2)
        cat_att = cat_q.matmul(cat_k)
        cat_att_map = torch.softmax(cat_att, dim=-1)
        r_v = r.view(b, self.dim, -1)
        d_v = d.view(b, self.dim, -1)

        r_att = r_v.matmul(cat_att_map).view(b, c, h, w)
        d_att = d_v.matmul(cat_att_map).view(b, c, h, w)
        final = cat + self.alpha * r_att + self.beta * d_att

        return F.interpolate(self.conv_final(self.gamma * final + add), scale_factor=self.div)


# region balance module
class RBM(nn.Module):
    def __init__(self, channel):
        super(RBM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv = conv3x3(channel * 2, channel)
        self.ra_f = RA(dim=channel)
        self.ra_d = RA(dim=channel)

    def forward(self, r, d, label=None):
        add = r + d
        mul = r * self.sigmoid(d)
        fuse = self.conv(torch.cat([add, mul], dim=1))

        d_ra = d * self.ra_d(d, label)
        fuse_ra = fuse * self.ra_f(fuse, label)
        return fuse_ra + d_ra


class Fusion1(nn.Module):
    def __init__(self, channel, another):
        super(Fusion1, self).__init__()
        self.conv1 = conv1x1(another, channel)
        self.conv2 = conv3x3(channel * 2, channel)
        self.ca = ChannelAttention(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, r, d):
        line1 = r + d
        line2 = r * self.sigmoid(d)
        line_fuse = self.conv2(torch.cat([line1, line2], dim=1))
        fuse_att1 = self.ca(line_fuse)
        line_fuse = line_fuse * fuse_att1
        return line_fuse


# ChannelInformationModule
class CIM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=3, padding=1):
        super(CIM, self).__init__()

        self.fc1 = Conv2d(in_planes, in_planes // ratio, kernel_size=kernel_size, padding=padding, bias=False)
        self.fc2 = Conv2d(in_planes // ratio, in_planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.ca = ChannelAttention(in_planes)

    def forward(self, x):
        out = self.ca(x * self.sigmoid(self.fc2(self.fc1(x))))
        return out


# Inter-and-intra balance module
class IIBNet(nn.Module):
    def __init__(self, n_classes=41):
        super(IIBNet, self).__init__()

        resnet_raw_model1 = convnext_base(pretrained=True, in_22k=True)
        resnet_raw_model2 = convnext_base(pretrained=True, in_22k=True)
        filters = [128, 256, 512, 1024]

        # # encoder for rgb image # #
        self.r_downsample_layers1 = resnet_raw_model1.downsample_layers[0]
        self.r_stages1 = resnet_raw_model1.stages[0]
        self.r_downsample_layers2 = resnet_raw_model1.downsample_layers[1]
        self.r_stages2 = resnet_raw_model1.stages[1]
        self.r_downsample_layers3 = resnet_raw_model1.downsample_layers[2]
        self.r_stages3 = resnet_raw_model1.stages[2]
        self.r_downsample_layers4 = resnet_raw_model1.downsample_layers[3]
        self.r_stages4 = resnet_raw_model1.stages[3]

        # # encoder for depth image # #
        self.d_downsample_layers1 = resnet_raw_model2.downsample_layers[0]
        self.d_stages1 = resnet_raw_model2.stages[0]
        self.d_downsample_layers2 = resnet_raw_model2.downsample_layers[1]
        self.d_stages2 = resnet_raw_model2.stages[1]
        self.d_downsample_layers3 = resnet_raw_model2.downsample_layers[2]
        self.d_stages3 = resnet_raw_model2.stages[2]
        self.d_downsample_layers4 = resnet_raw_model2.downsample_layers[3]
        self.d_stages4 = resnet_raw_model2.stages[3]

        self.low_fem = FEM(filters[0], 64, div=8)
        self.high_fem = FEM(filters[3], 64, div=8)
        # self.high_fem = Fusion1(filters[3], filters[3])

        self.fuse_module1 = RBM(filters[0])
        self.fuse_module2 = RBM(filters[1])
        self.fuse_module3 = RBM(filters[2])
        self.fuse_module4 = RBM(filters[3])

        self.ca1 = CIM(filters[0], kernel_size=9, padding=4)
        self.ca2 = CIM(filters[1], kernel_size=7, padding=3)
        self.ca3 = CIM(filters[2], kernel_size=5, padding=2)
        self.ca4 = CIM(filters[3], kernel_size=3, padding=1)
        self.cah = CIM(filters[3], kernel_size=3, padding=1)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()

        self.conv1 = conv1x1(filters[0], filters[1])
        self.conv2 = conv1x1(filters[0], filters[2])
        self.conv3 = conv1x1(filters[0], filters[3])

        self.conv_cat0 = conv3x3(filters[0]*2, filters[0])
        self.conv_cat1 = conv3x3(filters[1]*2, filters[1])
        self.conv_cat2 = conv3x3(filters[2]*2, filters[2])
        self.conv_cat3 = conv3x3(filters[3]*2, filters[3])

        self.m4 = M1(filters[3], filters[2])
        self.m3 = M(filters[2], filters[1])
        self.m2 = M(filters[1], filters[0])
        self.m1 = M(filters[0], filters[0]//2)

        self.conv64_nclass = conv3x3(int(filters[0]/2), n_classes)
        self.conv128_nclass = conv3x3(filters[0], n_classes)
        self.conv256_nclass = conv3x3(filters[1], n_classes)
        self.conv512_nclass = conv3x3(filters[2], n_classes)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, rgb, depth, label):
        # # Encoder # #
        rgb = self.r_downsample_layers1(rgb)
        rgb1 = self.r_stages1(rgb)
        depth = self.d_downsample_layers1(depth)
        depth1 = self.d_stages1(depth)

        rgb2 = self.r_downsample_layers2(rgb1)
        rgb2 = self.r_stages2(rgb2)
        depth2 = self.d_downsample_layers2(depth1)
        depth2 = self.d_stages2(depth2)

        rgb3 = self.r_downsample_layers3(rgb2)
        rgb3 = self.r_stages3(rgb3)
        depth3 = self.d_downsample_layers3(depth2)
        depth3 = self.d_stages3(depth3)

        rgb4 = self.r_downsample_layers4(rgb3)
        rgb4 = self.r_stages4(rgb4)
        depth4 = self.d_downsample_layers4(depth3)
        depth4 = self.d_stages4(depth4)

        fuse1 = self.fuse_module1(rgb1, depth1, label)
        fuse2 = self.fuse_module2(rgb2, depth2, label)
        fuse3 = self.fuse_module3(rgb3, depth3, label)
        fuse4 = self.fuse_module4(rgb4, depth4, label)

        # CIIB
        hweight4 = self.high_fem(rgb4, depth4)

        fuse1_ca = self.ca1(fuse1)
        fuse2_ca = self.ca2(fuse2)
        fuse3_ca = self.ca3(fuse3)
        fuse4_ca = self.ca4(fuse4)
        high_ca = self.cah(hweight4)

        fuse1_ca = fuse1_ca.repeat_interleave(8, dim=1)
        fuse2_ca = fuse2_ca.repeat_interleave(4, dim=1)
        fuse3_ca = fuse3_ca.repeat_interleave(2, dim=1)
        fuse4_ca = fuse4_ca

        # SIIB
        lweight1 = self.low_fem(rgb1, depth1)

        fuse1 = self.conv_cat0(torch.cat([fuse1, lweight1 * self.sa1(lweight1)], 1))
        satt1 = F.interpolate(self.conv1(lweight1), (60, 80))
        fuse2 = self.conv_cat1(torch.cat([fuse2, satt1 * self.sa2(satt1)], 1))
        satt2 = F.interpolate(self.conv2(lweight1), (30, 40))
        fuse3 = self.conv_cat2(torch.cat([fuse3, satt2 * self.sa3(satt2)], 1))
        satt3 = F.interpolate(self.conv3(lweight1), (15, 20))
        fuse4 = self.conv_cat3(torch.cat([fuse4, satt3 * self.sa4(satt3)], 1))

        top4 = self.m4(fuse4)
        top3 = self.m3(fuse3, top4)
        top2 = self.m2(fuse2, top3)
        top1 = self.m1(fuse1, top2)

        sal_out = self.up2(self.conv64_nclass(top1))
        out2 = F.interpolate(self.conv128_nclass(top2), scale_factor=4)
        out3 = F.interpolate(self.conv256_nclass(top3), scale_factor=8)
        out4 = F.interpolate(self.conv512_nclass(top4), scale_factor=16)

        return sal_out, out2, out3, out4, fuse1_ca, fuse2_ca, fuse3_ca, fuse4_ca, high_ca


class RA(nn.Module):
    def __init__(self, dim, num_classes=41):
        super(RA, self).__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(dim, dim//16, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim//16, dim, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def _dequeue_and_enqueue(self, keys, labels):
        batch_size, feat_dim, H, W = keys.size()
        segment_queue = torch.randn(batch_size, self.num_classes, feat_dim)

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)  # (c, hw)
            this_label = labels[bs].contiguous().view(-1)  # hw
            this_label_ids = torch.unique(this_label)  # [0-40]
            this_label_ids = [x for x in this_label_ids]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)  # 对类的某个区域的值求平均 shape: [dim]
                segment_queue[bs, lb, :] = feat
        return segment_queue

    def forward(self, t_feats, labels=None):
        labels.detach()
        b, c, h, w = t_feats.shape
        labels = labels.unsqueeze(1).float()
        labels = F.interpolate(labels, (t_feats.shape[2], t_feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()

        ori_t_fea = t_feats
        ori_labels = labels

        att = self._dequeue_and_enqueue(ori_t_fea, ori_labels).cuda()
        att = F.normalize(att, dim=1)
        att = torch.sum(att, dim=1).view(b, c, 1, 1)
        att = self.conv2(self.relu(self.conv1(att)))
        return self.sigmoid(att)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = conv1x1(in_planes, in_planes // ratio)
        self.relu1 = nn.ReLU()
        self.fc2 = conv1x1(in_planes // ratio, in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

