"""Swin Transformer模型加载模块

这个模块负责加载和配置原始的Swin Transformer模型。
目前支持的模型：
1. Swin-L/12/384: 大型Swin Transformer，窗口大小为12，输入分辨率为384x384
"""

import timm

from .swin_common import _make_swin_backbone


def _make_pretrained_swinl12_384(pretrained, hooks=None):
    """创建预训练的Swin-L/12/384模型
    
    Args:
        pretrained (bool): 是否使用预训练权重
        hooks (list, optional): 特征提取的hook点。默认为[1, 1, 17, 1]
            - 第1层：第1个block
            - 第2层：第1个block
            - 第3层：第17个block
            - 第4层：第1个block
    
    Returns:
        nn.Module: 配置好的Swin Transformer backbone
        
    说明：
        模型配置：
        - 架构：Swin-Large
        - patch大小：4x4
        - 窗口大小：12x12
        - 输入分辨率：384x384
    """
    # 创建原始的Swin Transformer模型
    model = timm.create_model("swin_large_patch4_window12_384", pretrained=pretrained)

    # 设置特征提取点
    hooks = [1, 1, 17, 1] if hooks == None else hooks
    
    # 构建backbone
    return _make_swin_backbone(
        model,
        hooks=hooks
    )
