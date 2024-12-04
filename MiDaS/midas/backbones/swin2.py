"""Swin Transformer V2模型加载模块

这个模块负责加载和配置Swin Transformer V2模型。
支持的模型变体：
1. Swin V2-Large：大型模型，动态窗口大小(12->24)，分辨率384x384
2. Swin V2-Base：基础模型，动态窗口大小(12->24)，分辨率384x384
3. Swin V2-Tiny：轻量级模型，固定窗口大小16，分辨率256x256

特点：
- 支持动态窗口大小
- 支持分辨率调整
- 预训练权重来自ImageNet-22K和ImageNet-1K
"""

import timm

from .swin_common import _make_swin_backbone


def _make_pretrained_swin2l24_384(pretrained, hooks=None):
    """创建预训练的Swin V2-Large模型
    
    Args:
        pretrained (bool): 是否使用预训练权重
        hooks (list, optional): 特征提取的hook点。默认为[1, 1, 17, 1]
            - 第1层：第1个block
            - 第2层：第1个block
            - 第3层：第17个block
            - 第4层：第1个block
    
    Returns:
        nn.Module: 配置好的Swin V2 backbone
        
    说明：
        模型配置：
        - 架构：Swin V2-Large
        - 窗口大小：从12到24动态调整
        - 分辨率：从192到384动态调整
        - 预训练：ImageNet-22K -> ImageNet-1K
    """
    # 创建原始的Swin V2-Large模型
    model = timm.create_model("swinv2_large_window12to24_192to384_22kft1k", pretrained=pretrained)

    # 设置特征提取点
    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )


def _make_pretrained_swin2b24_384(pretrained, hooks=None):
    """创建预训练的Swin V2-Base模型
    
    Args:
        pretrained (bool): 是否使用预训练权重
        hooks (list, optional): 特征提取的hook点。默认为[1, 1, 17, 1]
            - 第1层：第1个block
            - 第2层：第1个block
            - 第3层：第17个block
            - 第4层：第1个block
    
    Returns:
        nn.Module: 配置好的Swin V2 backbone
        
    说明：
        模型配置：
        - 架构：Swin V2-Base
        - 窗口大小：从12到24动态调整
        - 分辨率：从192到384动态调整
        - 预训练：ImageNet-22K -> ImageNet-1K
    """
    # 创建原始的Swin V2-Base模型
    model = timm.create_model("swinv2_base_window12to24_192to384_22kft1k", pretrained=pretrained)

    # 设置特征提取点
    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )


def _make_pretrained_swin2t16_256(pretrained, hooks=None):
    """创建预训练的Swin V2-Tiny模型
    
    Args:
        pretrained (bool): 是否使用预训练权重
        hooks (list, optional): 特征提取的hook点。默认为[1, 1, 5, 1]
            - 第1层：第1个block
            - 第2层：第1个block
            - 第3层：第5个block（注意：与大型模型不同）
            - 第4层：第1个block
    
    Returns:
        nn.Module: 配置好的Swin V2 backbone
        
    说明：
        模型配置：
        - 架构：Swin V2-Tiny
        - 窗口大小：固定16x16
        - 输入分辨率：256x256
        - patch网格：64x64
    """
    # 创建原始的Swin V2-Tiny模型
    model = timm.create_model("swinv2_tiny_window16_256", pretrained=pretrained)

    # 设置特征提取点（注意：第3层使用5而不是17）
    hooks = [1, 1, 5, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks,
        patch_grid=[64, 64]  # 为Tiny模型设置较小的patch网格
    )
