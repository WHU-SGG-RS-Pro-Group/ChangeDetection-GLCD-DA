import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Residual3(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, H,W,**kwargs):
        return self.fn(x, H,W, **kwargs) + x

class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)
class PreNorm3(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, H,W, **kwargs):
        return self.fn(self.norm(x), H,W, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # # 前馈神经网络 = 2个全连接层
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):  #dim=32   #dim： 输入和输出维度heads：  多头自注意力的头的数目 dim_head：  每个头的维度
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
       #self.scale = dim ** -0.5  #
        self.scale = dim_head ** -0.5
        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads  #h:8  _32 b:8 n:4096=64*64
        q = self.to_q(x)#x:torch.Size([8, 4096, 32]) q:torch.Size([8, 4096, 64])
        k = self.to_k(m)#m:torch.Size([8, 4, 32]) k:torch.Size([8, 4, 64])
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])  #k、v:torch.Size([8, 8, 4, 8]) q：torch.Size([8, 8, 4096, 8])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale #dots：torch.Size([8, 8, 4096, 4])
        mask_value = -torch.finfo(dots.dtype).max  #float数值

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)  #attn：torch.Size([8, 8, 4096, 4])
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  #out(bhid)=attn(bhij)v(bhjd)  out：torch.Size([8, 8, 4096, 8])
        out = rearrange(out, 'b h n d -> b n (h d)') #out：torch.Size([8, 4096, 64])
        out = self.to_out(out)
        # vis_tmp2(out)

        return out

class Cross_AgentAttention(nn.Module):
    def __init__(self, dim, num_patches,featuresize=4096,heads=8,dim_head = 64, proj_drop=0.,qkv_bias=True, qk_scale=None, attn_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by heads {heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        window_size2 = (int(featuresize ** 0.5), int(featuresize ** 0.5))
        self.window_size2= window_size2
        self.heads = heads
        self.dim_head=dim_head
        inner_dim = dim_head * heads
        self.inner_dim=inner_dim
        # head_dim = dim // heads
        self.scale = dim_head ** -0.5

        self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, inner_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, heads, window_size2[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, heads, 1, window_size2[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, m, mask = None):
        b, n, c = x.shape#X1
        _, n2, _,= m.shape#TOKEN1
        H = int(n  ** 0.5)
        W = int(n  ** 0.5)
        H2 = int(n2 ** 0.5)
        W2 = int(n2 ** 0.5)

        heads = self.heads
        # head_dim = c // heads
        head_dim=self.dim_head
        q = self.q(x)  #(8,4096,512)

        if self.sr_ratio > 1:
            m_ = m.permute(0, 2, 1).reshape(b, c, H, W)
            m_ = self.sr(m_).reshape(b, c, -1).permute(0, 2, 1)
            m_ = self.norm(m_)
            kv = self.kv(m_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(m)
            kv = self.kv(m).reshape(b, -1, 2, self.inner_dim).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]#(k,v 8,4,512=8*64)

        agent_tokens = self.pool(q.reshape(b, H, W, self.inner_dim).permute(0, 3, 1, 2)).reshape(b, self.inner_dim, -1).permute(0, 2, 1)#(8,49,512)
        q = q.reshape(b, n, heads, head_dim).permute(0, 2, 1, 3)#(8,8,4096,64)
        k = k.reshape(b, n2 // self.sr_ratio ** 2, heads, head_dim).permute(0, 2, 1, 3)#(8,8,4,64)
        v = v.reshape(b, n2 // self.sr_ratio ** 2, heads, head_dim).permute(0, 2, 1, 3)#(8,8,4,64)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, heads, head_dim).permute(0, 2, 1, 3)#(8,8,49,64)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)#(2,2)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, heads, self.agent_num, -1).repeat(b, 1, 1, 1)#(8,8,49,4)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, heads, self.agent_num, -1).repeat(b, 1, 1, 1)#(8,8,49,4)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)#(8,8,49,4)
        agent_attn = self.attn_drop(agent_attn)#(8,8,49,4)
        agent_v = agent_attn @ v#(8,8,49,64)

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window_size2[0],self.window_size2[1]), mode='bilinear')#(8,49,64,64)
        agent_bias1 = agent_bias1.reshape(1, heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)#(8,8,4096,49)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, heads, -1, self.agent_num).repeat(b, 1, 1, 1)#(8,8,4096,49)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)#(8,8,4096,49)
        q_attn = self.attn_drop(q_attn)#(8,8,4096,49)
        x = q_attn @ agent_v#(8,8,4096,64)

        x = x.transpose(1, 2).reshape(b, n, self.inner_dim)#(8,8,4096,512)
        # v = v.transpose(1, 2).reshape(b, H2 // self.sr_ratio, W2 // self.sr_ratio, self.inner_dim).permute(0, 3, 1, 2)
        # if self.sr_ratio > 1:
        #     v = nn.functional.interpolate(v, size=(H2, W2), mode='bilinear')
        # x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, self.inner_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        #self.scale = dim ** -0.5#缩放因子
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        # chunk: qkv tuple
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #qkv
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # q * k转置 除以根号d_k
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, heads=8,dim_head = 64, proj_drop=0.,qkv_bias=True, qk_scale=None, attn_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by heads {heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.heads = heads
        self.dim_head=dim_head
        inner_dim = dim_head * heads
        self.inner_dim=inner_dim
        # head_dim = dim // heads
        self.scale = dim_head ** -0.5

        self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, inner_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,H,W):

        b, n, c = x.shape
        heads = self.heads
        # head_dim = c // heads
        head_dim=self.dim_head
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x)
            kv = self.kv(x).reshape(b, -1, 2, self.inner_dim).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, self.inner_dim).permute(0, 3, 1, 2)).reshape(b, self.inner_dim, -1).permute(0, 2, 1)
        q = q.reshape(b, n, heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, self.inner_dim)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, self.inner_dim).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, self.inner_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Residual3(PreNorm3(dim, AgentAttention(dim,num_patches, heads = heads, dim_head = dim_head, proj_drop = dropout))),#更改后注意力
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),#一般注意力模块
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x,mask = None):
        for attn, ff in self.layers:
            # x = attn(x, mask = mask)
            # x = attn(x,H,W)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),

                # Residual2(PreNorm2(dim, Cross_AgentAttention(dim, num_patches,featuresize, heads = heads,
                #                                         dim_head = dim_head, proj_drop = dropout,attn_drop=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
                # Residual(PreNorm(dim, FeedForward(dim, mlp_dim, bias=True)))

            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x

class SA(nn.Module):
    def __init__(self, nFeat):
        super(SA, self).__init__()
        self.conv1_sa = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.conv2_sa = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=3, stride=1, padding=1, bias=True), nn.Sigmoid())
        self.conv3_sa = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.conv1_sa(x)
        weight_sa = self.conv2_sa(out1)#(8,1,64,64)
        out = self.conv3_sa(torch.mul(x, weight_sa))
        return out
class CA(nn.Module):
    def __init__(self, nFeat, ratio=8):
        super(CA, self).__init__()
        self.conv1_ca = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.conv2_ca = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.avg_pool_ca = nn.AdaptiveAvgPool2d(1)
        self.max_pool_ca = nn.AdaptiveMaxPool2d(1)
        self.fc1_ca = nn.Sequential(
            nn.Conv2d(nFeat, nFeat//ratio, kernel_size=1, padding=0, bias=True), nn.PReLU())
        self.fc2_ca = nn.Sequential(
            nn.Conv2d(nFeat//ratio, nFeat, kernel_size=1, padding=0, bias=True), nn.Sigmoid())
        self.conv_avgmax = nn.Conv2d(2*nFeat, nFeat, kernel_size=1,
                                  padding=0, bias=False)
    def forward(self, x):
        out1 = self.conv1_ca(x)
        avg_pool=self.avg_pool_ca(out1)
        max_pool=self.max_pool_ca(out1)
        out1=self.conv_avgmax(torch.cat([avg_pool, max_pool], dim=1))
        #avg_weight_ca = self.fc2_ca(self.fc1_ca(self.avg_pool_ca(out1)))
        avg_weight_ca = self.fc2_ca(self.fc1_ca(out1))
        out = self.conv2_ca(torch.mul(x, avg_weight_ca))
        return out
class RCSA(nn.Module):
    def __init__(self, nFeat):
        super(RCSA, self).__init__()
        self.ca_rcsa = CA(nFeat)
        self.sa_rcsa = SA(nFeat)
        self.conv1_rcsa = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.conv1_rcsa(torch.add(self.ca_rcsa(x), self.sa_rcsa(x)))
        out = torch.add(x, out1)
        return out

class Classifier(nn.Module):
    def __init__(self, in_chan=32, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))

    def forward(self, x):
        x = self.head(x)
        return x




