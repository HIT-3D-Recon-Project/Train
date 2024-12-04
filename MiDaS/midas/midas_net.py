"""MiDaS网络：通过混合多个数据集训练的单目深度估计网络
这个文件包含了改编自以下代码的实现：
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py

主要特点：
1. 使用ResNeXt101作为backbone
2. 采用特征融合块进行多尺度特征融合
3. 支持非负深度输出
"""

import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, Interpolate, _make_encoder


class MidasNet(BaseModel):
    """MiDaS单目深度估计网络
    
    这个网络通过混合多个数据集进行训练，能够在各种场景下进行鲁棒的深度估计。
    网络架构包括：
    1. ResNeXt101主干网络用于特征提取
    2. 4个特征融合块用于多尺度特征融合
    3. 输出卷积层用于生成最终的深度图
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """初始化MiDaS网络
        
        Args:
            path (str, optional): 预训练模型的路径。默认为None
            features (int, optional): 特征维度。默认为256
            non_negative (bool, optional): 是否确保输出为非负值。默认为True
        """
        print("正在加载权重: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        # 创建ResNeXt101主干网络
        self.pretrained, self.scratch = _make_encoder(
            backbone="resnext101_wsl", 
            features=features, 
            use_pretrained=use_pretrained
        )

        # 创建4个特征融合块，用于融合不同尺度的特征
        self.scratch.refinenet4 = FeatureFusionBlock(features)  # 最深层的特征融合
        self.scratch.refinenet3 = FeatureFusionBlock(features)  # 第三层特征融合
        self.scratch.refinenet2 = FeatureFusionBlock(features)  # 第二层特征融合
        self.scratch.refinenet1 = FeatureFusionBlock(features)  # 最浅层的特征融合

        # 创建输出卷积层，用于生成最终的深度图
        self.scratch.output_conv = nn.Sequential(
            # 将特征通道数从features降到128
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            # 上采样特征图
            Interpolate(scale_factor=2, mode="bilinear"),
            # 将特征通道数从128降到32
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # 最后输出单通道深度图
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # 如果需要非负输出，则使用ReLU
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        # 如果提供了预训练模型路径，则加载模型
        if path:
            self.load(path)

    def forward(self, x):
        """前向传播函数
        
        Args:
            x (tensor): 输入数据（图像）
                形状: [batch_size, 3, height, width]
        
        Returns:
            tensor: 预测的深度图
                形状: [batch_size, height, width]
        
        处理流程：
        1. 通过ResNeXt101提取4个不同尺度的特征
        2. 对这些特征进行重新映射
        3. 通过4个特征融合块自底向上融合特征
        4. 最后通过输出卷积层生成深度图
        """
        # 1. 通过ResNeXt101提取特征
        layer_1 = self.pretrained.layer1(x)        # 1/4分辨率
        layer_2 = self.pretrained.layer2(layer_1)  # 1/8分辨率
        layer_3 = self.pretrained.layer3(layer_2)  # 1/16分辨率
        layer_4 = self.pretrained.layer4(layer_3)  # 1/32分辨率

        # 2. 特征重新映射
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 3. 特征融合（自底向上）
        path_4 = self.scratch.refinenet4(layer_4_rn)               # 开始于最深层
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)      # 融合path_4和layer_3
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)      # 融合path_3和layer_2
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)      # 融合path_2和layer_1

        # 4. 生成深度图
        out = self.scratch.output_conv(path_1)

        # 移除深度维度并返回
        return torch.squeeze(out, dim=1)
