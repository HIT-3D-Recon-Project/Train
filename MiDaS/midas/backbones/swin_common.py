"""Swin Transformer公共模块

这个模块提供了Swin Transformer系列模型的公共功能：
1. 前向传播函数
2. Backbone构建函数
3. 特征后处理操作

主要用于处理Swin Transformer的层级特征提取和重组织，支持：
1. 原始Swin Transformer
2. Swin Transformer V2
"""

import torch
import torch.nn as nn
import numpy as np

from .utils import activations, forward_default, get_activation, Transpose


def forward_swin(pretrained, x):
    """Swin Transformer的前向传播函数
    
    使用默认的前向传播方式处理Swin Transformer模型。
    
    Args:
        pretrained: 预训练的Swin Transformer模型
        x: 输入数据
        
    Returns:
        tuple: 多个层次的特征图
    """
    return forward_default(pretrained, x)


def _make_swin_backbone(
        model,
        hooks=[1, 1, 17, 1],
        patch_grid=[96, 96]
):
    """构建Swin Transformer backbone
    
    将Swin Transformer模型转换为特征提取backbone，主要步骤：
    1. 注册钩子函数以获取中间特征
    2. 创建特征后处理操作
    3. 处理特征图的空间结构
    
    Args:
        model: 原始的Swin Transformer模型
        hooks (list): 每层中需要提取特征的block索引，默认为[1, 1, 17, 1]
        patch_grid (list): 输入patch的网格大小，默认为[96, 96]
        
    Returns:
        nn.Module: 配置好的backbone模型
        
    注意：
        特征图大小在不同层级上依次缩小：
        - 第1层：原始patch_grid大小
        - 第2层：patch_grid // 2
        - 第3层：patch_grid // 4
        - 第4层：patch_grid // 8
    """
    pretrained = nn.Module()

    # 设置模型并注册钩子
    pretrained.model = model
    pretrained.model.layers[0].blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.layers[1].blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.layers[2].blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.layers[3].blocks[hooks[3]].register_forward_hook(get_activation("4"))

    # 存储激活值
    pretrained.activations = activations

    # 获取patch网格大小
    if hasattr(model, "patch_grid"):
        used_patch_grid = model.patch_grid
    else:
        used_patch_grid = patch_grid

    patch_grid_size = np.array(used_patch_grid, dtype=int)

    # 为每个层级创建后处理操作
    # 第1层：原始分辨率
    pretrained.act_postprocess1 = nn.Sequential(
        Transpose(1, 2),  # 调整维度顺序
        nn.Unflatten(2, torch.Size(patch_grid_size.tolist()))  # 重组为2D特征图
    )
    
    # 第2层：1/2分辨率
    pretrained.act_postprocess2 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 2).tolist()))
    )
    
    # 第3层：1/4分辨率
    pretrained.act_postprocess3 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 4).tolist()))
    )
    
    # 第4层：1/8分辨率
    pretrained.act_postprocess4 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 8).tolist()))
    )

    return pretrained
