"""
MiDaS模型构建块模块
这个文件包含了MiDaS模型中使用的各种神经网络构建块，包括编码器、特征融合、残差卷积等组件
"""

import torch
import torch.nn as nn

# 导入各种backbone网络的预训练模型和前向传播函数
from .backbones.beit import (
    _make_pretrained_beitl16_512,
    _make_pretrained_beitl16_384,
    _make_pretrained_beitb16_384,
    forward_beit,
)
from .backbones.swin_common import (
    forward_swin,
)
from .backbones.swin2 import (
    _make_pretrained_swin2l24_384,
    _make_pretrained_swin2b24_384,
    _make_pretrained_swin2t16_256,
)
from .backbones.swin import (
    _make_pretrained_swinl12_384,
)
from .backbones.levit import (
    _make_pretrained_levit_384,
    forward_levit,
)
from .backbones.vit import (
    _make_pretrained_vitb_rn50_384,
    _make_pretrained_vitl16_384,
    _make_pretrained_vitb16_384,
    forward_vit,
)

def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True, hooks=None,
                  use_vit_only=False, use_readout="ignore", in_features=[96, 256, 512, 1024]):
    """创建编码器网络
    
    Args:
        backbone (str): 主干网络类型，如'beitl16_512'、'swin2l24_384'等
        features (int): 特征维度
        use_pretrained (bool): 是否使用预训练模型
        groups (int): 卷积分组数
        expand (bool): 是否扩展特征通道
        exportable (bool): 是否可导出
        hooks (list): 用于提取特征的钩子函数
        use_vit_only (bool): 是否仅使用ViT部分
        use_readout (str): readout类型
        in_features (list): 输入特征维度列表
    
    Returns:
        tuple: (预训练模型, scratch模型)
    """
    if backbone == "beitl16_512":
        pretrained = _make_pretrained_beitl16_512(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # BEiT_512-L (backbone)
    elif backbone == "beitl16_384":
        pretrained = _make_pretrained_beitl16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # BEiT_384-L (backbone)
    elif backbone == "beitb16_384":
        pretrained = _make_pretrained_beitb16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # BEiT_384-B (backbone)
    elif backbone == "swin2l24_384":
        pretrained = _make_pretrained_swin2l24_384(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [192, 384, 768, 1536], features, groups=groups, expand=expand
        )  # Swin2-L/12to24 (backbone)
    elif backbone == "swin2b24_384":
        pretrained = _make_pretrained_swin2b24_384(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [128, 256, 512, 1024], features, groups=groups, expand=expand
        )  # Swin2-B/12to24 (backbone)
    elif backbone == "swin2t16_256":
        pretrained = _make_pretrained_swin2t16_256(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # Swin2-T/16 (backbone)
    elif backbone == "swinl12_384":
        pretrained = _make_pretrained_swinl12_384(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [192, 384, 768, 1536], features, groups=groups, expand=expand
        )  # Swin-L/12 (backbone)
    elif backbone == "next_vit_large_6m":
        from .backbones.next_vit import _make_pretrained_next_vit_large_6m
        pretrained = _make_pretrained_next_vit_large_6m(hooks=hooks)
        scratch = _make_scratch(
            in_features, features, groups=groups, expand=expand
        )  # Next-ViT-L on ImageNet-1K-6M (backbone)
    elif backbone == "levit_384":
        pretrained = _make_pretrained_levit_384(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [384, 512, 768], features, groups=groups, expand=expand
        )  # LeViT 384 (backbone)
    elif backbone == "vitl16_384":
        pretrained = _make_pretrained_vitl16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # ViT-L/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb_rn50_384":
        pretrained = _make_pretrained_vitb_rn50_384(
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
        )
        scratch = _make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )  # ViT-H/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb16_384":
        pretrained = _make_pretrained_vitb16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # ViT-B/16 - 84.6% Top1 (backbone)
    elif backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)  # efficientnet_lite3
    elif backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
        
    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    """创建scratch网络
    
    Args:
        in_shape (list): 输入特征维度列表
        out_shape (int): 输出特征维度
        groups (int): 卷积分组数
        expand (bool): 是否扩展特征通道
    
    Returns:
        nn.Module: scratch网络
    """
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False):
    """创建预训练的efficientnet_lite3网络
    
    Args:
        use_pretrained (bool): 是否使用预训练模型
        exportable (bool): 是否可导出
    
    Returns:
        nn.Module: 预训练的efficientnet_lite3网络
    """
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable
    )
    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    """创建efficientnet_backbone网络
    
    Args:
        effnet (nn.Module): efficientnet网络
    
    Returns:
        nn.Module: efficientnet_backbone网络
    """
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained
    

def _make_resnet_backbone(resnet):
    """创建resnet_backbone网络
    
    Args:
        resnet (nn.Module): resnet网络
    
    Returns:
        nn.Module: resnet_backbone网络
    """
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    """创建预训练的resnext101_wsl网络
    
    Args:
        use_pretrained (bool): 是否使用预训练模型
    
    Returns:
        nn.Module: 预训练的resnext101_wsl网络
    """
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)



class Interpolate(nn.Module):
    """插值模块
    用于上采样或下采样特征图
    """
    
    def __init__(self, scale_factor, mode, align_corners=False):
        """初始化插值模块
        
        Args:
            scale_factor (float): 缩放因子
            mode (str): 插值模式，如'bilinear'、'nearest'等
            align_corners (bool): 是否对齐角点
        """
        super(Interpolate, self).__init__()
        
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """前向传播
        
        Args:
            x (tensor): 输入特征图
            
        Returns:
            tensor: 插值后的特征图
        """
        return self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

class ResidualConvUnit(nn.Module):
    """残差卷积单元
    包含两个3x3卷积层和ReLU激活函数，并有残差连接
    """
    
    def __init__(self, features):
        """初始化残差卷积单元
        
        Args:
            features (int): 特征通道数
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (tensor): 输入特征图
            
        Returns:
            tensor: 经过残差卷积处理的特征图
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        return out + x  # 残差连接

class FeatureFusionBlock(nn.Module):
    """特征融合块
    用于融合不同层次的特征
    """
    
    def __init__(self, features):
        """初始化特征融合块
        
        Args:
            features (int): 特征通道数
        """
        super(FeatureFusionBlock, self).__init__()
        
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        
    def forward(self, *xs):
        """前向传播
        
        Args:
            *xs: 可变数量的输入特征图
            
        Returns:
            tensor: 融合后的特征图
        """
        output = xs[0]
        
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
            
        output = self.resConfUnit2(output)
        
        return output

class ResidualConvUnit_custom(nn.Module):
    """残差卷积单元
    包含两个3x3卷积层和ReLU激活函数，并有残差连接
    """
    
    def __init__(self, features, activation, bn):
        """初始化残差卷积单元
        
        Args:
            features (int): 特征通道数
            activation (nn.Module): 激活函数
            bn (bool): 是否使用批归一化
        """
        super().__init__()
        
        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """前向传播
        
        Args:
            x (tensor): 输入特征图
            
        Returns:
            tensor: 经过残差卷积处理的特征图
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """特征融合块
    用于融合不同层次的特征
    """
    
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """初始化特征融合块
        
        Args:
            features (int): 特征通道数
            activation (nn.Module): 激活函数
            deconv (bool): 是否使用反卷积
            bn (bool): 是否使用批归一化
            expand (bool): 是否扩展特征通道
            align_corners (bool): 是否对齐角点
            size (tuple): 输出特征图大小
        """
        super(FeatureFusionBlock_custom, self).__init__()
        
        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """前向传播
        
        Args:
            *xs: 可变数量的输入特征图
            size (tuple): 输出特征图大小
            
        Returns:
            tensor: 融合后的特征图
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
