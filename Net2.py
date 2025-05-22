import torch.nn.functional as F
import torch
from deformabled import DeformableConv2d
import torch.nn as nn
from block import Block
from timm.models.layers import to_2tuple



class SLKA(nn.Module):
    def __init__(self, n_feats, act_layer=nn.GELU, shrink=0.5, scale=2):
        super().__init__()
        f = int(n_feats * shrink)
        self.head = nn.Conv2d(n_feats, f, 1)
        self.proj_2 = nn.Conv2d(f, f, kernel_size=3)
        self.activation = nn.GELU()
        self.LKA = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=7, padding=7 // 2),
            self.activation,
            nn.Conv2d(f, f, kernel_size=5, padding=5 // 2),
            self.activation,
            nn.Conv2d(f, f, kernel_size=5, padding=5 // 2),
            nn.Sigmoid()
        )
        self.LKA_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(f, f, kernel_size=7, padding=7 // 2),
            self.activation,
            nn.Conv2d(f, f, kernel_size=5, padding=5 // 2),
            self.activation,
            nn.Conv2d(f, f, kernel_size=5, padding=5 // 2),
            nn.Sigmoid()
        )
        self.tail = nn.Conv2d(f, n_feats, 1)
        self.Dconv = DeformableConv2d(f, f, 5, padding=2)
        self.scale = scale


    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        c1 = self.Dconv(self.head(x))
        c2 = F.max_pool2d(c1, kernel_size=self.scale * 2 + 1, stride=self.scale)
        c2 = self.LKA(c2)
        c3 = F.interpolate(c2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        a = self.tail(c3 + self.LKA_2(c1))
        a = F.sigmoid(a)
        return (x * a).reshape(B, C, -1).permute(0, 2, 1)



class Embed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        # _, _, H, W = x.shape
        if self.norm is not None:
            x = self.norm(x)
        return x

class Merge(nn.Module):
    def __init__(self, dim, h, w):
        super(Merge, self).__init__()
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=2, stride=2, padding=0)
        self.h = h
        self.dim = dim
        self.w = w
        self.norm = nn.BatchNorm2d(dim*2)

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
        x = self.norm(self.conv(x))

        return x.reshape(B, self.dim*2, -1).permute(0, 2, 1)

class Expand(nn.Module):
    def __init__(self, dim, h):
        super(Expand, self).__init__()
        self.dim = dim
        self.h = h
        self.conv = nn.ConvTranspose2d(self.dim, self.dim//2, 2, stride=2)

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.h)
        x = self.conv(x)

        return x.reshape(B, self.dim//2, -1).permute(0, 2, 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = Embed(512)

        self.l1 = nn.Sequential(Block(96, 128, 4, 8, 3),
                                Block(96, 128, 4, 8, 3))

        self.l2 = nn.Sequential(Block(192, 64, 4, 8, 3),
                                Block(192, 64, 4, 8, 3))

        self.l3 = nn.Sequential(Block(384, 32, 4, 8, 3),
                                Block(384, 32, 4, 8, 3))

        self.l4 = nn.Sequential(Block(768, 16, 4, 8, 3),
                                Block(768, 16, 4, 8, 3),
                                Block(768, 16, 4, 8, 3),
                                Block(768, 16, 4, 8, 3))

        self.m1 = Merge(96, 128, 128)
        self.m2 = Merge(192, 64, 64)
        self.m3 = Merge(384, 32, 32)

        self.p3 = Expand(768, 16)
        self.p2 = Expand(384, 32)
        self.p1 = Expand(192, 64)

        self.d3 = nn.Sequential(Block(384, 32, 4, 8, 3),
                                Block(384, 32, 4, 8, 3))

        self.d2 = nn.Sequential(Block(192, 64, 4, 8, 3),
                                Block(192, 64, 4, 8, 3))

        self.d1 = nn.Sequential(Block(96, 128, 4, 8, 3),
                                Block(96, 128, 4, 8, 3))

        self.dbm3 = SLKA(384)
        self.dbm2 = SLKA(192)
        self.dbm1 = SLKA(96)

        self.up = nn.PixelShuffle(4)
        self.seg = nn.Conv2d(6, 1, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.embed(x) #torch.Size([1, 16384, 96])
        x1 = self.l1(x) #torch.Size([1, 16384, 96])
        #
        x = self.m1(x1) #torch.Size([1, 4096, 192])
        x2 = self.l2(x) #torch.Size([1, 4096, 192])
        #
        x = self.m2(x2) #torch.Size([1, 1024, 384])
        x3 = self.l3(x) #torch.Size([1, 1024, 384])
        #
        x = self.m3(x3) #torch.Size([1, 256, 768])
        x4 =self.l4(x) #torch.Size([1, 256, 768])
        #
        x = self.p3(x4) #torch.Size([1, 1024, 384])
        x3_temp = self.dbm3(x3+x)
        x = self.d3(x3_temp) #torch.Size([1, 1024, 384])
        #
        x = self.p2(x) #torch.Size([1, 4096, 192])
        x2_temp = self.dbm2(x2+x)
        x = self.d2(x2_temp)
        #
        x = self.p1(x) #torch.Size([1, 16384, 96])
        x1_temp = self.dbm1(x1+x)
        x = self.d1(x1_temp) #128x128
        #
        x = self.up(x.permute(0, 2, 1).reshape(B, 96, 128, 128)) #torch.Size([1, 6, 512, 512])
        x = self.seg(x)
        return x

if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512).cuda()
    # y = torch.rand(1, 1024, 384).cuda()
    # dbm = DeBlock(384).cuda()
    part = Net().cuda()
    out = part(x)
    print(out.shape)
    # out = dbm(y)
    # print(out.shape)