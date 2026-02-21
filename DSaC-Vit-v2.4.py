# 相较论文代码模型 进行部分修正1

from torchinfo import summary
import torch.nn.functional as F
from thop import profile
import torch
from torch import nn
import random
import numpy as np


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.scaling_num = 2

    def forward(self, x):
        # print(x.size)
        # x: input features with shape [b, c, h, w]
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print("ceshi ECa")
        y = self.sigmoid(y)
        # print(y.size)
        #print(y==y.expand_as(x))
        x = x * y.expand_as(x)
        # print(y)
        return x


class channel_attention(nn.Module):
    def __init__(self, input_channel, ratio=2):
        super(channel_attention, self).__init__()
        self.Average_pool = nn.AdaptiveAvgPool2d(1)
        self.Max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        ###############################################
        self.conv1d = nn.Conv1d(input_channel, input_channel, kernel_size=2,groups=input_channel, bias=False)
        ###############################################应该用分组卷积
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.bn2 = nn.BatchNorm2d(input_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        Average_out = self.sigmoid(self.bn1((self.conv(self.Average_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))))
        Max_out = self.sigmoid(self.bn2((self.conv(self.Max_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))))
        # print("Max_out",Max_out.size())
        # print("Average_out",Average_out.size())
        out = torch.cat((Max_out, Average_out), dim=2).squeeze(3)
        #out = self.sigmoid(out)
        print("out", out.size())
        out = self.conv1d(out).unsqueeze(3)
        print("out", out.size())
        return out


class space_attention(nn.Module):
    def __init__(self):
        super(space_attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3,
                              padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        Average_out = torch.mean(x, dim=1, keepdim=True)
        Max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(Max_out.size(), Average_out.size())
        out = self.conv(self.sigmoid(torch.cat([Max_out, Average_out], dim=1)))
        #out = self.sigmoid(out)

        # print(avg_out.size())
        return out


class BTUM(nn.Module):  # Bootstrap the upsampling module
    def __init__(self):
        super(BTUM, self).__init__()
        self.ca = channel_attention(64, 2)
        self.sa = space_attention()
        self.SiLu = nn.SiLU()
        # self.conv1 = nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x1, x2):
        ca_out = self.ca(x2)
        sa_out = self.sa(x2)
        # print("ca_out",ca_out.size())
        # print("sa_out",sa_out.size())
        out_att = ca_out * sa_out
        out_att = self.bn1(out_att)
        out_att = self.SiLu(out_att)
        print("out_att", out_att.size())
        # print(out.size())
        # print("out_att",out_att.size())
        # x1 = self.conv1(x1)
        # print("x1",x1.size())
        x1 = F.interpolate(x1, size=(11, 11), mode='bilinear', align_corners=True)
        print("上采样特征尺寸", x1.size())
        # print("x1",x1.size())
        mix_out = x1 * out_att
        mix_out = self.bn2(mix_out)
        mix_out = self.SiLu(mix_out)
        return mix_out


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class conv1x1qkv_block(nn.Module):
    def __init__(self, in_planes=64, out_planes=64):
        super(conv1x1qkv_block, self).__init__()
        self.convq = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.convk = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.convv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        q, k, v = self.convq(x), self.convk(x), self.convv(x)
        return q, k, v


class UnfoldBlock(nn.Module):
    def __init__(self, in_planes=64, out_planes=64, head_num=4, kernel_size=5, padding_att=1):
        super(UnfoldBlock, self).__init__()

        self.wise = kernel_size
        self.kernel_size = kernel_size - 4
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=1)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.padding_att = padding_att
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=1)
        self.norm = nn.BatchNorm2d(in_planes).cuda()

    def forward(self, q, k, v):
        b = q.shape[0]
        # print("q,k,v",q.shape,k.shape,v.shape)
        C = self.in_planes * self.kernel_size * self.kernel_size
        N = ((self.wise - self.kernel_size + 2 * self.padding_att) + 1) * (
                (self.wise - self.kernel_size + 2 * self.padding_att) + 1)
        pos_embed = nn.Parameter(torch.zeros(1, N, C).cuda())
        # print("pos_embed", pos_embed.shape)
        cls_tokenq = nn.Parameter(torch.zeros(1, 1, C).cuda())
        # print("cls_tokenq", cls_tokenq.shape)
        cls_tokenq = cls_tokenq.expand(b, -1, -1)
        # print("cls_tokenq", cls_tokenq.shape)

        cls_tokenk = nn.Parameter(torch.zeros(1, 1, C).cuda())
        # print("cls_tokenk", cls_tokenk.shape)
        cls_tokenk = cls_tokenk.expand(b, -1, -1)
        # print("cls_tokenk", cls_tokenk.shape)

        cls_tokenv = nn.Parameter(torch.zeros(1, 1, C).cuda())
        # print("cls_tokenv", cls_tokenv.shape)
        cls_tokenv = cls_tokenv.expand(b, -1, -1)
        # print("cls_tokenv", cls_tokenv.shape)

        q_pad = self.pad_att(q)
        # print("q_pad", q_pad.shape)
        unfold_q = self.unfold(q_pad).permute(0, 2, 1) + pos_embed
        # print("unfold_q", unfold_q.shape)

        unfold_q = torch.cat([cls_tokenq, unfold_q], dim=1)
        # print("unfold_q", unfold_q.shape)

        k_pad = self.pad_att(k)
        # print("k_pad", k_pad.shape)
        unfold_k = self.unfold(k_pad).permute(0, 2, 1) + pos_embed
        # print("unfold_k", unfold_k.shape)
        unfold_k = torch.cat([cls_tokenk, unfold_k], dim=1)
        # print("unfold_k", unfold_k.shape)

        v_pad = self.pad_att(v)
        # print("v_pad", v_pad.shape)
        unfold_v = self.unfold(v_pad).permute(0, 2, 1) + pos_embed
        # print("unfold_v", unfold_v.shape)
        unfold_v = torch.cat([cls_tokenv, unfold_v], dim=1)
        # print("unfold_v", unfold_v.shape)

        return unfold_q, unfold_k, unfold_v


##############################################################
# ConvBlock:输入经过conv1*1的qkv得到经过卷积的特征，展开
##############################################################


class ConvBlock(nn.Module):
    def __init__(self, out_planes=64, in_planes=64, kernel_conv=5, head_num=4):
        super(ConvBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_conv = kernel_conv
        self.head_num = head_num
        self.head_dim = out_planes // head_num
        self.fc = nn.Conv2d(3 * self.head_num, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False).cuda()
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=0,
                                  stride=1).cuda()
        self.bn = nn.BatchNorm2d(in_planes)
        self.reset_parameters()

    def reset_parameters(self):
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x, q, k, v):
        b, c, h, w = x.shape
        # print("x.shape",x.shape)
        # print(b,c,h,w)
        size = h
        f_all = self.fc(torch.cat(
            [q.view(b, self.head_num, self.head_dim, h * w), k.view(b, self.head_num, self.head_dim, h * w),
             v.view(b, self.head_num, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        # print("f_conv", f_all.permute(0, 2, 1, 3).shape)
        # print("f_conv", f_conv.shape)

        out_conv = self.dep_conv(f_conv)
        out_conv = self.bn(out_conv)
        # print("out_conv", out_conv.shape)
        out_conv = out_conv.reshape(b, self.in_planes, -1)
        out_conv = out_conv.reshape(b, self.in_planes, size - 4, size - 4)
        # print("out_conv", out_conv.shape)
        return out_conv


##############################################################
# Attention
##############################################################


class Attention(nn.Module):
    def __init__(self, i, head_num=4, in_planes=64):
        super(Attention, self).__init__()
        self.i = i
        self.unfold = {}
        self.head_num = head_num
        self.head_dim = in_planes // head_num
        self.att_drop = nn.Dropout(0.1)
        self.in_planes = in_planes

        self.convblock1 = ConvBlock()
        self.unfold[1] = UnfoldBlock(kernel_size=15)
        self.unfold[2] = UnfoldBlock(kernel_size=13)
        self.qkv1 = conv1x1qkv_block()
        self.conv_global = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False)

    def forward(self, x1):

        if self.i == 1:
            size = 15
        else:
            size = 13
        q, k, v = torch.tensor([]).cuda(), torch.tensor([]).cuda(), torch.tensor([]).cuda()
        out_conv = torch.tensor([]).cuda()
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        q_conv, k_conv, v_conv = self.qkv1(x1)
        # print("conv", q_conv.shape, k_conv.shape, v_conv.shape)
        q_temp, k_temp, v_temp = self.unfold[self.i](q_conv, k_conv, v_conv)
        # print("temp", q_temp.shape, k_temp.shape, v_temp.shape)
        q = torch.cat([q, q_temp], dim=2)
        k = torch.cat([k, k_temp], dim=2)
        v = torch.cat([v, v_temp], dim=2)
        # print("q, k, v", q.shape, k.shape, v.shape)
        conv = self.convblock1(x1, q_conv, k_conv, v_conv)
        # print("conv", conv.shape)
        out_conv = torch.cat([out_conv, conv], dim=1)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        b, N, C = q.shape
        print(b, N, C)
        qurry = q.reshape(b, self.head_num, N, C // self.head_num)
        key = k.reshape(b, self.head_num, N, C // self.head_num)
        value = v.reshape(b, self.head_num, N, C // self.head_num)
        # print("qurry", qurry.shape)
        # print("key", key.shape)
        # print("value", value.shape)
        # print("qurry, key, value", qurry.shape, key.shape, value.shape)

        scaling = float(self.head_dim) ** -0.5
        att = (qurry @ key.transpose(-2, -1)) * scaling
        # att = self.att_drop(att)
        # print(att.shape)
        out_attn = (att @ value).transpose(1, 2).reshape(b, N, C)
        # print((att @ value).transpose(1, 2).shape)
        out_attn = out_attn[:, 1, :].reshape(b, self.in_planes, size - 4, size - 4)
        out_attn = self.conv_global(out_attn)
        # print("out_attn",out_attn.shape)

        rate1 = torch.nn.Parameter(torch.Tensor(1)).cuda()
        rate2 = torch.nn.Parameter(
            torch.randn(out_attn.shape[0], out_attn.shape[1], out_attn.shape[2], out_attn.shape[3])).cuda()
        # print("rate2",rate2.size())
        init_rate_half(rate1)
        init_rate_half(rate2)
        # print(out_attn.shape)
        # print(self.rate1)
        return out_conv * rate1 + out_attn * rate2


class MPAcV(nn.Module):
    def __init__(self, i):
        super(MPAcV, self).__init__()
        self.i = i
        self.Attention = Attention(self.i)

    def forward(self, x1):
        # print(self.i)
        out = self.Attention(x1)
        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, i, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = MPAcV(i)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("out", out.shape)  #out torch.Size([32, 64, 4, 4])
        # print("~~~~~~~~~~~~~~~~~~out~~~~~~~~~~~~~~~~~~~",out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out = self.relu(out)
        return out


class HyperBCS_2D(nn.Module):

    def __init__(self, block=Bottleneck, input_channels=28, num_classes=11,
                 groups=1, norm_layer=None):
        super(HyperBCS_2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(self.inplanes)
        self.bn3 = norm_layer(self.inplanes)
        self.bn4 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool_multiscale = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.layer1 = Bottleneck(inplanes=64, planes=64, i=1)
        self.layer2 = Bottleneck(inplanes=64, planes=64, i=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.scale1_rate = torch.nn.Parameter(torch.Tensor(1)).cuda()
        self.scale2_rate = torch.nn.Parameter(torch.Tensor(1)).cuda()
        self.conv_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.ECA = eca_layer(64)
        self.Btum = BTUM()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv2_yd = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.scale1_rate)
        init_rate_half(self.scale2_rate)

#2025.7.16 和ppt对应，逐个conv+bn+relu
    def forward(self, x):
        # print(x.shape)
        # print("INDIAN~")
        # print("模型循环~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        x = x.squeeze(1)
        x = self.ECA(x)

        # print("x{},x_multiscale{}".format(x.shape, x_multiscale.shape))# [32, 28, 8, 8],[32, 28, 4, 4]
        # print(x.shape)

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        print("x1", x1.shape)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        print("x2", x2.shape)


        x2_yd = self.conv2_yd(x2)
        print("x2_yd", x2_yd.shape)

        # print(x0.shape)
        # print("conv1",x.shape)
        # print("x{},x_multiscale{}".format(x.shape, x_multiscale.shape))[32, 64, 8, 8]，[32, 64, 4, 4]
        # print("x{},x_multiscale{}".format(x.shape, x_multiscale.shape))[32, 64, 8, 8]，[32, 64, 4, 4]


        # print("""x1{},x2{}""".format(x1.shape, x2.shape))
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        print("x1_经过PSCViT", x1.shape)
        print("x2_经过PSCViT", x2.shape)


        # print("x1的形状:{},x2的形状{}".format(x1.shape,x2.shape))
        x_btum = self.Btum(x2, x2_yd)
        # print("x",x.shape)
        x_total = torch.cat((x1, x_btum), dim=1)
        x_total = self.conv_fusion(x_total)
        print("最终融合特征F",x_total.shape)
        x_total = self.bn4(x_total)
        x_total = self.relu(x_total)
        x_total = self.avgpool(x_total)
        # x_multiscale = self.avgpool(x_multiscale)
        # print("x_total",x_total.shape)
        x_total = torch.flatten(x_total, 1)
        # print("x_total",x_total.shape)
        # x_multiscale = torch.flatten(x_multiscale, 1)

        x_total = self.fc(x_total)
        # print("x",x.shape)
        return x_total


if __name__ == '__main__':
    model = HyperBCS_2D(input_channels=35).cuda()
    input = torch.randn([32, 35, 15, 15]).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(model(input).shape)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}M".format(flops / 1e6))
    print("params:{:.3f}M".format(params / 1e6))
    # --------------------------------------------------#
    #   用来测试网络能否跑通，同时可查看FLOPs和params
    # --------------------------------------------------#
    summary(model, input_size=(32, 35, 15, 15))
