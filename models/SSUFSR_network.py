# from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from torch.distributions.uniform import Uniform
import numpy as np
import random


def create_model(args):
    return SSUFSRNet(args)


class SSUFSRNet(nn.Module):
    def __init__(self, args):
        super(SSUFSRNet, self).__init__()

        # Params  
        n_feats = args.n_feats  ### 16 ###  
        self.scale = args.scale  
        self.window_sizes = [8, 16]  
        self.n_blocks = args.n_blocks  

        self.head = nn.Conv2d(args.colors, n_feats, kernel_size=3, bias=True, stride=1, padding=1, padding_mode='reflect')
        
        self.body = nn.ModuleList(  
            [SSRM(in_channels=n_feats, out_channels=n_feats, num_heads=4, bs=8, ks=3, sr=2, scale=1.0, ratio=1.0) for i in range(args.n_blocks)]
        )  
        
        self.moe = MOE(nf=n_feats)  
        
        if self.scale == 4:
            self.tail= nn.Sequential(  
                nn.Conv2d(n_feats, n_feats*4, kernel_size=1, bias=True, stride=1, padding=0, padding_mode='reflect'),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(n_feats, n_feats*4, kernel_size=1, bias=True, stride=1, padding=0, padding_mode='reflect'),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(n_feats, 3, kernel_size=3, bias=True, stride=1, padding=1, padding_mode='reflect'),
            )
        else:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats*self.scale*self.scale, kernel_size=1, bias=True, stride=1, padding=0, padding_mode='reflect'),
                nn.PixelShuffle(self.scale),
                nn.GELU(),
                nn.Conv2d(n_feats, 3, kernel_size=3, bias=True, stride=1, padding=1, padding_mode='reflect'),
            )
            
    def forward(self, x):
        
        H, W = (x.shape[2], x.shape[3])
        x = self.check_image_size(x)
        
        res = self.head(x) 
        
        a = res
        
        exp = []
        for blkid in range(self.n_blocks):
            a = self.body[blkid](a, res)
            if (blkid+1) % (self.n_blocks // 3) == 0 or (blkid+1) == self.n_blocks:
                exp.append(a)
        
        a = self.moe(exp) + res  
        
        a = self.tail(a) + F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)  

        # a = torch.clamp(a, min=0.0, max=self.rgb_range)
        
        return a[:, :, 0:H*self.scale, 0:W*self.scale]  

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # wsize = self.window_sizes
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class MOE(nn.Module):
    def __init__(self, nf=32, dropout_ratio=0.0):
        super(MOE, self).__init__()
        self.nf = nf  
        self.dropout_ratio = dropout_ratio  
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        
        self.fc_a = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.fc_b = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.fc_c = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.fuse = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        self.softmax=nn.Softmax(dim=0)
        
    def forward(self, x):
        
        if len(x) == 4:  
            a, b, c = x[1], x[2], x[3]  
        else:
            a, b, c = x[0], x[1], x[2]  
        
        wa = self.fc_a(a)  
        wb = self.fc_b(b)  
        wc = self.fc_c(c)
        # wa = self.fc_a(torch.mean(a, dim=[2,3], keepdim=True))  
        # wb = self.fc_b(torch.mean(b, dim=[2,3], keepdim=True))  
        # wc = self.fc_c(torch.mean(c, dim=[2,3], keepdim=True))  
        wm = self.softmax(torch.stack([wa, wb, wc], dim=0))
        
        a = a * wm[0,:].squeeze(0)  
        b = b * wm[1,:].squeeze(0)  
        c = c * wm[2,:].squeeze(0)  
        
        out = a + b + c  
        
        if self.dropout_ratio > 0:  
            out = self.dropout(out)  
        
        out = self.fuse(out) + out    
        
        return out


class SSRM(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, bs=8, ks=3, sr=2, scale=2.0, ratio=1):
        super(SSRM, self).__init__()  

        self.norm = LayerNorm(dim=in_channels, LayerNorm_type='WithBias')  ## WithBias/BiasFree/AffineFree  
        self.s1 = MEM(dim=in_channels, out_dim=out_channels, scale=scale, bias=True, ks=ks, ratio=ratio)  
        self.s2 = MEM(dim=in_channels, out_dim=out_channels, scale=scale, bias=True, ks=ks, ratio=ratio)
        self.s3 = ESA(ch=in_channels, block_size=bs, halo_size=1, num_heads=num_heads, bias=False, ks=3, sr=sr)
        self.s4 = MEM(dim=in_channels, out_dim=out_channels, scale=scale, bias=True, ks=ks, ratio=ratio) 
        self.ffn = MEM(dim=in_channels, out_dim=out_channels, scale=scale, bias=True, ks=ks, ratio=ratio)  
        
    def forward(self, a, y):  
        # norm
        a = self.norm(a)  
        
        # s1  
        z = self.s1(a)  
        # z = F.softshrink(z, lambd=0.1)  
        
        # s2  
        b = self.s2(a)  
        
        # s3  
        v = a + z + y  
        v = self.s3(v) + v  
        
        # s4
        a = v - b  
        # a = F.softshrink(a, lambd=0.1)  
        a = self.s4(a) + b  
        
        ##
        a = self.ffn(a) + a  
        
        return a


class CA(nn.Module):
    def __init__(self, nf, reduction=8, res=True):
        super(CA, self).__init__()
        self.is_res = res
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf*1, int(nf // reduction), 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(nf // reduction), nf*1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x
        ca = self.attn(x)
        
        if self.is_res:
            return out.mul(1 + ca)
        else:
            return out.mul(ca)


class ESA(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=3, num_heads=4, bias=False, ks=3, sr=1):
        super(ESA, self).__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_ch = ch // num_heads
        assert ch % num_heads == 0, "ch should be divided by # heads"
        
        self.sr = sr
        if sr > 1:
            self.sampler = nn.MaxPool2d(2, sr)
            self.LocalProp = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=ks, stride=1, padding=int(ks//2), groups=ch, bias=True, padding_mode='reflect'),
                Interpolate(scale_factor=sr, mode='bilinear', align_corners=True),
            )

        self.rel_h = nn.Parameter(torch.randn(1, block_size+2*halo_size, 1, self.head_ch//2), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(1, 1, block_size+2*halo_size, self.head_ch//2), requires_grad=True)
        self.qkv_conv = nn.Conv2d(ch, ch*3, kernel_size=1, bias=bias)
        # self.weight = nn.Parameter(torch.ones(1, ch, 1, 1), requires_grad=True)
        # self.conv = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0, groups=1, bias=True, padding_mode='reflect')

    def forward(self, x):
        
        b, c, oh, ow = x.size()
        
        if self.sr > 1:
            x = self.sampler(x)
        
        # pad feature maps to multiples of window size  
        B, C, H, W = x.size()
        pad_l = pad_t = 0
        pad_r = (self.block_size - W % self.block_size) % self.block_size
        pad_b = (self.block_size - H % self.block_size) % self.block_size
        
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='reflect')  

        b, c, h, w, block, halo, heads = *x.shape, self.block_size, self.halo_size, self.num_heads
        assert h % block == 0 and w % block == 0, 'feature map dimensions must be divisible by the block size'
        
        x = self.qkv_conv(x)
        
        q, k, v = torch.chunk(x, 3, dim=1)

        # q = q * self.conv(self.weight)
        q = rearrange(q, 'b c (h k1) (w k2) -> (b h w) (k1 k2) c', k1=block, k2=block)
        q = q * self.head_ch ** -0.5  # b*#blocks, flattened_query, c

        k = F.unfold(k, kernel_size=block+halo*2, stride=block, padding=halo)
        k = rearrange(k, 'b (c a) l -> (b l) a c', c=c)

        v = F.unfold(v, kernel_size=block+halo*2, stride=block, padding=halo)
        v = rearrange(v, 'b (c a) l -> (b l) a c', c=c)
        # v = v + torch.mean(v, dim=1, keepdim=True)
        # v = v * torch.sigmoid(torch.mean(v, dim=1, keepdim=True)) + v

        # b*#blocks*#heads, flattened_vector, head_ch  
        q, v = map(lambda i: rearrange(i, 'b a (h d) -> (b h) a d', h=heads), (q, v))
        # positional embedding  
        k = rearrange(k, 'b (k1 k2) (h d) -> (b h) k1 k2 d', k1=block+2*halo, h=heads)
        k_h, k_w = k.split(self.head_ch//2, dim=-1)  
        k = torch.cat([k_h+self.rel_h, k_w+self.rel_w], dim=-1)  
        k = rearrange(k, 'b k1 k2 d -> b (k1 k2) d')  
        
        # b*#blocks*#heads, flattened_query, flattened_neighborhood  
        sim = torch.einsum('b i d, b j d -> b i j', q, k)  
        attn = F.softmax(sim, dim=-1)  
        
        # b*#blocks*#heads, flattened_query, head_ch
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h w n) (k1 k2) d -> b (n d) (h k1) (w k2)', b=b, h=(h//block), w=(w//block), k1=block, k2=block)
        
        if self.sr > 1:
            out = self.LocalProp(out)
        
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :oh, :ow]

        return out


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups, padding_mode='reflect',
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        
        return m
    

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn * x + x

    
class MEM(nn.Module):
    def __init__(self, dim, out_dim, scale, bias, ks=3, ratio=1):
        super(MEM, self).__init__()
        
        ## V4
        self.project_in1 = nn.Conv2d(dim, int(dim*scale), kernel_size=1, groups=1, bias=bias) 
        self.project_in2 = nn.Sequential(
            nn.Conv2d(dim, int(dim*ratio), kernel_size=1, groups=1, bias=bias),
            nn.Conv2d(int(dim*ratio), int(dim*ratio), kernel_size=ks, stride=1, padding=int(ks//2), groups=int(dim*ratio//1), bias=True, padding_mode='reflect'),
            nn.Conv2d(int(dim*ratio), int(dim*scale), kernel_size=1, groups=1, bias=bias),
        )
        self.project_out = nn.Conv2d(int(dim*scale), out_dim, kernel_size=1, bias=bias)
        self.act = nn.GELU()  
    
    def forward(self, x):
        
        x1 = self.project_in1(x)
        x2 = self.project_in2(x)
        
        x2 = self.act(x1) * x2
        
        x = self.project_out(x2)
        
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners,
                          recompute_scale_factor=True)
        return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = 1e-6

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + self.eps) * self.weight


class AffineFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(AffineFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.normalized_shape = normalized_shape
        self.eps = 1e-6

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + self.eps)
    

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
        self.eps = 1e-6

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type =='AffineFree':
            self.body = AffineFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    