"""
Basic SSPT model.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.ops import DeformConv2dPack


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, Hx, Wx, z=None, Hz=None, Wz=None):
        if z is not None:
            B, Nx, C = x.shape
            B, Nz, C = z.shape

            qx = self.q(x).reshape(B, Nx, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            qz = self.q(z).reshape(B, Nz, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            if not self.linear:
                if self.sr_ratio > 1:
                    x_ = x.permute(0, 2, 1).reshape(B, C, Hx, Wx)
                    x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                    x_ = self.norm(x_)
                    kvx = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

                    z_ = z.permute(0, 2, 1).reshape(B, C, Hz, Wz)
                    z_ = self.sr(z_).reshape(B, C, -1).permute(0, 2, 1)
                    z_ = self.norm(z_)

                    kvz = self.kv(z_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                else:
                    kvx = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    kvz = self.kv(z).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, Hx, Wx)
                x_= self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1) # [b, 49, c]
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kvx = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

                z_ = z.permute(0, 2, 1).reshape(B, C, Hz, Wz)
                z_ = self.sr(self.pool(z_)).reshape(B, C, -1).permute(0, 2, 1)  # [b, 49, c]
                z_ = self.norm(z_)
                z_ = self.act(z_)
                kvz = self.kv(z_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kx, vx = kvx[0], kvx[1]

            kz, vz = kvz[0], kvz[1]

            attnx = (qx @ kz.transpose(-2, -1)) * self.scale
            attnx = attnx.softmax(dim=-1)
            attnx = self.attn_drop(attnx)

            attnz = (qz @ kx.transpose(-2, -1)) * self.scale
            attnz = attnz.softmax(dim=-1)
            attnz = self.attn_drop(attnz)

            x = (attnx @ vz).transpose(1, 2).reshape(B, Nx, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            z = (attnz @ vx).transpose(1, 2).reshape(B, Nz, C)
            z = self.proj(z)
            z = self.proj_drop(z)
            return x, z
        else:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            if not self.linear:
                if self.sr_ratio > 1:
                    x_ = x.permute(0, 2, 1).reshape(B, C, Hx, Wx)
                    x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                    x_ = self.norm(x_)
                    kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                else:
                    kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, Hx, Wx)
                x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)  # [b, 49, c]
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, Hx, Wx, z=None, Hz=None, Wz=None, cross_flag = False):
        if cross_flag:
            x_reconst_from_z, z_reconst_from_x = self.attn(self.norm1(x), Hx, Wx, self.norm1(z), Hz, Wz)

            x_fuse = x + self.drop_path(x_reconst_from_z)
            z_fuse = z + self.drop_path(z_reconst_from_x)

            x = x_fuse + self.drop_path(self.mlp(self.norm2(x_fuse), Hx, Wx))
            z = z_fuse + self.drop_path(self.mlp(self.norm2(z_fuse), Hz, Wz))

        else:
            if z is not None:
                x = x + self.drop_path(self.attn(self.norm1(x), Hx, Wx))
                x = x + self.drop_path(self.mlp(self.norm2(x), Hx, Wx))

                z = z + self.drop_path(self.attn(self.norm1(z), Hz, Wz))
                z = z + self.drop_path(self.mlp(self.norm2(z), Hz, Wz))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), Hx, Wx))
                x = x + self.drop_path(self.mlp(self.norm2(x), Hx, Wx))

        return x, z


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class SSPT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6],
                 sr_ratios=[8, 4, 2], num_stages=3, linear=False, pretrained=None, cross_dict= None, cross_num=5, down_sample=[4, 2, 2, 2]):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        self.embed_dim = embed_dims[num_stages-1]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=down_sample[i],
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        #self.class_head = MLP_head(embed_dims[num_stages-1], embed_dims[num_stages-1], 1 + 1, 2)
        #self.bbox_head = MLP_head(embed_dims[num_stages-1], embed_dims[num_stages-1], 4, 2)
       # self.exchange_weight = nn.Parameter(torch.ones([2, cross_num]))

        self.cross_dict = cross_dict
        for i in range(len(self.cross_dict)):
            if len(self.cross_dict[i]) > 0:
               self.start_stage = i
               self.start_blk = self.cross_dict[i][0] -1
               break

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        return None

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


    def forward_single(self, x):
        x = self.forward_features(x)
        return x

    def forward_cross(self, x, z ):
       # exchange_weight = self.exchange_weight# nn.Parameter(torch.ones([cross_num, 2]))

        B = x.shape[0]
        outs = []
        stages = self.num_stages
        cross_index = 0

        seq_out = {"x":[],"z":[]}
        for i in range(stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, Hx, Wx = patch_embed(x)
            z, Hz, Wz = patch_embed(z)


            cnt=0
            for blk in block:
                cnt = cnt + 1
                if cnt in self.cross_dict[i]:
                    cross = True
                    cross_index = cross_index + 1
                    x, z = blk(x, Hx, Wx, z, Hz, Wz, cross_flag=cross )
                else:
                    cross = False
                    x, z = blk(x, Hx, Wx, z, Hz, Wz, cross_flag=cross)


            x = x.reshape(B, Hx, Wx, -1).permute(0, 3, 1, 2).contiguous()
            z = z.reshape(B, Hz, Wz, -1).permute(0, 3, 1, 2).contiguous()

            seq_out["x"].append(x)
            seq_out["z"].append(z)

        return seq_out

    def forward(self, x, z = None):
        x_fused = self.forward_cross(z, x)

        return x_fused




class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        # self.dwconv2 = DeformConv2dPack(dim, dim, kernel_size=3, stride=1, padding=3//2,deform_groups=2)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


class SSPT_base(SSPT):
    def __init__(self, **kwargs):
        super(SSPT_base, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320], num_heads=[1, 2, 5], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # depths=[2,2,8,3],
            depths=[3, 4, 10],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,down_sample=[4, 2, 2],  num_stages=3, cross_num=kwargs['cross_num'],cross_dict=[
            [],
            [],
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
            #[1,2,3],
        ])

    def load_param(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def load_param_self_backbone(self, pretrained=None):
        if isinstance(pretrained, str):
            model_dict = self.state_dict()
            state_dict = torch.load(pretrained)
            #model.load_state_dict(model_dict,strict=False)  # model加载dict中的数据，更新网络的初始值
            # state_dict = torch.load(pretrained)
            #  state_dict = {k.split("model_uav.transformer.")[-1]: v for k, v in state_dict.items() if
            #                (k.split("model_uav.transformer.")[-1] in model_dict)}
            self.load_state_dict(state_dict, strict=False)


def build_sspt(cross_dict=None):
    # b3: [3, 4, 10],
    if cross_dict is None:
        cross_dict = [
            [],
            [],
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
            # [1, 2, 3],
        ]
    model = SSPT_base(pretrained=False,  num_stages=3, cross_num=5)
    model_dict = model.state_dict()
    p = r'/media/zeh/4d723c17-52ed-4771-9ff5-c5b4cf1675e9/fjq/SSPT/pretrain/spvt_v2.pth'

    return model





if __name__ == '__main__':
    img_size = 384
    x = torch.rand(1, 3, 384, 384)
    z=torch.rand(1, 3, 128, 128)
    model = build_sspt()
    output=model(z,x)
