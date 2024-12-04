"""
DPT（Dense Prediction Transformers）深度估计模型实现
这个文件实现了基于Vision Transformer的深度估计模型DPT，它能够从单张图像中预测深度图
"""

import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_beit,
    forward_swin,
    forward_levit,
    forward_vit,
)
from .backbones.levit import stem_b4_transpose
from timm.models.layers import get_act_layer


def _make_fusion_block(features, use_bn, size = None):
    """创建特征融合块
    
    Args:
        features (int): 特征通道数
        use_bn (bool): 是否使用批归一化
        size (tuple): 输出特征图大小
        
    Returns:
        FeatureFusionBlock_custom: 特征融合块实例
    """
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPT(BaseModel):
    """DPT模型的基类
    实现了Vision Transformer的基本架构和特征提取功能
    """
    
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs
    ):
        """初始化DPT模型
        
        Args:
            head (nn.Module): 输出头部网络
            features (int): 特征维度
            backbone (str): 主干网络类型
            readout (str): 特征读取方式
            channels_last (bool): 是否将通道维度放在最后
            use_bn (bool): 是否使用批归一化
            **kwargs: 其他参数
        """
        super(DPT, self).__init__()

        self.channels_last = channels_last

        # 为不同类型的Transformer设置特征提取的位置
        # 对于Swin、Swin2、LeViT和Next-ViT这些层次化的Transformer，
        # hooks必须按照指定范围设置
        hooks = {
            "beitl16_512": [5, 11, 17, 23],
            "beitl16_384": [5, 11, 17, 23],
            "beitb16_384": [2, 5, 8, 11],
            "swin2l24_384": [1, 1, 17, 1],  # 允许范围: [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2b24_384": [1, 1, 17, 1],  # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2t16_256": [1, 1, 5, 1],   # [0, 1], [0,  1], [ 0,  5], [ 0,  1]
            "swinl12_384": [1, 1, 17, 1],   # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "next_vit_large_6m": [2, 6, 36, 39],  # [0, 2], [3,  6], [ 7, 36], [37, 39]
            "levit_384": [3, 11, 21],       # [0, 3], [6, 11], [14, 21]
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }[backbone]

        # 设置Next-ViT的输入特征维度
        if "next_vit" in backbone:
            in_features = {
                "next_vit_large_6m": [96, 256, 512, 1024],
            }[backbone]
        else:
            in_features = None

        # 实例化backbone并重组模块
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # 设置为True则从头训练，使用ImageNet权重
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        self.number_layers = len(hooks) if hooks is not None else 4
        size_refinenet3 = None
        self.scratch.stem_transpose = None

        # 根据backbone类型选择相应的前向传播函数
        if "beit" in backbone:
            self.forward_transformer = forward_beit
        elif "swin" in backbone:
            self.forward_transformer = forward_swin
        elif "next_vit" in backbone:
            from .backbones.next_vit import forward_next_vit
            self.forward_transformer = forward_next_vit
        elif "levit" in backbone:
            self.forward_transformer = forward_levit
            size_refinenet3 = 7
            self.scratch.stem_transpose = stem_b4_transpose(256, 128, get_act_layer("hard_swish"))
        else:
            self.forward_transformer = forward_vit

        # 创建特征融合网络
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head


    def forward(self, x):
        """前向传播
        
        Args:
            x (tensor): 输入图像张量
            
        Returns:
            tensor: 输出特征图
        """
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        if self.number_layers == 3:
            path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    """DPT深度估计模型
    继承自DPT基类，专门用于单目深度估计任务
    """
    
    def __init__(self, path=None, non_negative=True, **kwargs):
        """初始化DPT深度估计模型
        
        Args:
            path (str): 预训练模型路径
            non_negative (bool): 是否确保输出为非负值
            **kwargs: 其他参数，包括features、head_features_1、head_features_2等
        """
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        # 构建输出头部网络
        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),  # 如果需要非负输出则使用ReLU
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        # 如果提供了预训练模型路径，则加载模型
        if path is not None:
            self.load(path)

    def forward(self, x):
        """前向传播
        
        Args:
            x (tensor): 输入图像张量
            
        Returns:
            tensor: 预测的深度图
        """
        return super().forward(x).squeeze(dim=1)
