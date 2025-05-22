import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from Separable_convolution import S_conv
import math
import torch.nn.functional as F
from deformabled import DeformableConv2d
from einops import rearrange

class Spartial_Attention(nn.Module):  #空间注意机制在这里

    def __init__(self, kernel_size=9):
        super(Spartial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        y = x * mask
        return y

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 3, padding=1),
            nn.Sigmoid())
        self.attention_1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.attention(x)
        y2 = self.attention_1(x)
        out = self.sigmoid(y1+y2)*x
        return out

class CBAMBlock(nn.Module):
    def __init__(self, channel):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(num_feat=channel)
        self.sa = Spartial_Attention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad)

    def forward(self, x):
        return x * self.scale

class FastLeFF(nn.Module):

    def __init__(self, dim=32, bias=False, hidden_dim=128, num_heads=8, act_layer=nn.GELU, qk_norm=1, drop=0.):
        super().__init__()
        self.norm = qk_norm
        self.num_heads = num_heads
        self.drop = nn.Dropout(drop)
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),act_layer())
        self.dwconv1 = nn.Sequential(S_conv(hidden_dim, hidden_dim * 2), act_layer())
        self.dwconv3 = nn.Sequential(DeformableConv2d(hidden_dim, hidden_dim, 7, padding=3), act_layer())
        self.dwconv5 = nn.Sequential(DeformableConv2d(hidden_dim, hidden_dim, 5, padding=2), act_layer())
        self.dwconv4 = nn.Sequential(DeformableConv2d(hidden_dim, hidden_dim, 3), act_layer())
        self.dwconv2 = nn.Sequential(S_conv(hidden_dim, hidden_dim), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))


    def feat_decompose(self, x):  # MOGANET: MULTI-ORDER GATED AGGREGATION NETWORK
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x_1 = self.linear1(x)
        # spatial restore
        x_2 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32
        x1, x2 = self.dwconv1(x_2).chunk(2, dim=1)
        x1 = self.dwconv5(self.dwconv4(self.dwconv3(x1))) #这里为dwconv4
        x_3 = F.gelu(x1) * x2

        # flaten
        x_4 = rearrange(x_3, ' b c h w -> b (h w) c', h=hh, w=hh)
        x_5 = self.linear2(x_4)
        return x_5

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift_size=0, agent_num=49, **kwargs):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.shift_size = shift_size
        self.ca = CBAMBlock(dim)

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//16, kernel_size=5, padding=5 // 2),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(),
            nn.Conv2d(dim//16, dim, kernel_size=5, padding=5 // 2),
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        self.LKA = nn.Conv2d(dim, dim * 3, kernel_size=5, stride=1, padding=2, groups=dim)
        self.pool_1 = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.pool_2 = nn.AdaptiveMaxPool2d(output_size=(pool_size, pool_size))

    def forward(self, x, h, w,mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        x = self.LKA(x.reshape(b, h, w, c).permute(0, 3, 1, 2))
        #x = self.qkv(x)
        k, v, q = torch.chunk(x, 3, dim=1)  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c
        v_ = self.ca(self.conv5(v).reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)# 新增

        agent_tokens = self.pool_1(q).reshape(b, c, -1).permute(0, 2, 1)
        agent_tokens_k = self.pool_2(k).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v_1 = v_.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape) #torch.Size([1, 8, 16384, 16384])
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.attn_drop(self.softmax(attn))
        x_1 = (attn @ v).transpose(1, 2).reshape(b, n, c)

        attention_reshape = x_1.transpose(-2, -1).contiguous().view(b, c, h, w)
        channel_map = self.channel_interaction(attention_reshape)
        conv_x2 = torch.sigmoid(channel_map)*q.reshape(b, h, w, c).permute(0, 3, 1, 2)

        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens_k = agent_tokens_k.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2   #生成线性参数

        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn_1 = self.softmax((agent_tokens_k * self.scale) @ q.transpose(-2, -1) + position_bias)

        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v_1 + agent_attn_1 @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        k_attn = self.softmax((k * self.scale) @ agent_tokens_k.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        k_attn = self.attn_drop(k_attn)
        x = (q_attn+k_attn) @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)+self.dwc(conv_x2).permute(0, 2, 3, 1).reshape(b, n, c)
        q = q.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(q).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Block(nn.Module):
    def __init__(self, num_feat, input_resolution, window_size=8, heads=8, compress_ratio=3, mlp_ratio=4, drop=0., act_layer=nn.GELU, alpha=0.):
        super(Block, self).__init__()

        self.norm1 = nn.LayerNorm(num_feat)
        self.norm2 = nn.LayerNorm(num_feat)
        self.wsa = WindowAttention(num_feat, (window_size, window_size), heads)
        self.mlp = FastLeFF(num_feat)
        self.windows_size = window_size
        self.windows_num = (input_resolution / window_size) * (input_resolution / window_size)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)
        self.windows_num = (H * W) // (self.windows_size * self.windows_size)
        out = x.reshape(B, C, -1).permute(0, 2, 1)  # 新增
        w_x = out.reshape(B, H, W, C)#新增
        w_x = window_partition(w_x, self.windows_size)
        b, h, w, c = w_x.shape
        w_x = w_x.reshape(b, h * w, c)
        wsa_f = self.wsa(w_x, h, w)  # torch.Size([1024, 16, 96])
        res = int(self.windows_num ** 0.5)
        x_temp = wsa_f.reshape(B, res, res, self.windows_size, self.windows_size, C).permute(0, 5, 1, 3, 2, 4)
        wsa_f = x_temp.reshape(B, C, -1).permute(0, 2, 1)

        x_1 = self.norm1(wsa_f)#新调整
        x_2 = self.mlp(x_1+x.reshape(B, H * W, C))
        x_2 = self.norm2(x_2)

        return x.reshape(B, H * W, C)+x_2



if __name__ == '__main__':
    x = torch.rand(1, 16384, 96).cuda()
    # y = torch.rand(1, 128, 128, 96).cuda()
    # model = CAB(96, 3, 16).cuda()
    model = Block(96, 128, 4, 8, 3, 16, alpha=0.5).cuda()
    #model = Block(96, 128, 4, 8, 3, 16).cuda()
    # model = WindowAttention(96, (4, 4), 8).cuda()
    # out = model(x)
    # out = window_partition(y, 4).reshape(1024, 16, 96)
    out = model(x)
    print(out.shape)
