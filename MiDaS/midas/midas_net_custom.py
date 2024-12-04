"""MiDaS自定义网络模块：用于单目深度估计的网络，通过混合多个数据集进行训练。

这个模块实现了一个轻量级版本的MiDaS网络（MidasNet_small），主要特点：
1. 使用EfficientNet-Lite3作为默认backbone
2. 支持特征扩展（expand）选项
3. 可导出为其他格式（如ONNX）
4. 支持channels_last内存格式优化

代码部分改编自：
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""

import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, FeatureFusionBlock_custom, Interpolate, _make_encoder


class MidasNet_small(BaseModel):
    """轻量级单目深度估计网络
    
    这是MiDaS网络的一个更小、更快的变体，专门针对移动设备和实时应用进行了优化。
    主要特点：
    1. 使用轻量级backbone（默认为EfficientNet-Lite3）
    2. 可选的特征扩展机制
    3. 支持模型导出
    4. 内存布局优化选项
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}):
        """初始化网络
        
        Args:
            path (str, optional): 预训练模型的路径。默认为None
            features (int, optional): 基础特征通道数。默认为64
            backbone (str, optional): 编码器backbone网络。默认为efficientnet_lite3
            non_negative (bool, optional): 是否确保输出为非负值。默认为True
            exportable (bool, optional): 是否使网络可导出。默认为True
            channels_last (bool, optional): 是否使用channels_last内存格式。默认为False
            align_corners (bool, optional): 上采样时是否对齐角点。默认为True
            blocks (dict, optional): 网络块的配置选项。默认启用expand
        """
        print("正在加载权重: ", path)

        super(MidasNet_small, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        # 设置特征通道数
        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features        # 保持原始通道数
            features2=features*2      # 2倍通道数
            features3=features*4      # 4倍通道数
            features4=features*8      # 8倍通道数

        # 创建编码器
        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
  
        # 设置激活函数
        self.scratch.activation = nn.ReLU(False)    

        # 创建特征融合块
        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        # 创建输出卷积层
        self.scratch.output_conv = nn.Sequential(
            # 1. 降低通道数并上采样
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            # 2. 进一步处理特征
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            # 3. 生成最终深度图
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # 4. 确保非负（如果需要）
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        
        # 加载预训练权重（如果提供）
        if path:
            self.load(path)


    def forward(self, x):
        """前向传播
        
        Args:
            x (tensor): 输入数据（图像）

        Returns:
            tensor: 预测的深度图
            
        处理流程：
        1. 可选的内存格式优化
        2. 通过backbone提取多尺度特征
        3. 特征重组织
        4. 自底向上的特征融合
        5. 生成最终深度图
        """
        # 优化内存布局（如果启用）
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)

        # 1. 通过backbone提取特征
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        # 2. 特征重组织
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 3. 自底向上的特征融合
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        # 4. 生成深度图
        out = self.scratch.output_conv(path_1)
        
        return out


def fuse_model(m):
    prev_previous_type = nn.Identity()
    prev_previous_name = ''
    previous_type = nn.Identity()
    previous_name = ''
    for name, module in m.named_modules():
        if prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d and type(module) == nn.ReLU:
            # print("FUSED ", prev_previous_name, previous_name, name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name, name], inplace=True)
        elif prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d:
            # print("FUSED ", prev_previous_name, previous_name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name], inplace=True)
        # elif previous_type == nn.Conv2d and type(module) == nn.ReLU:
        #    print("FUSED ", previous_name, name)
        #    torch.quantization.fuse_modules(m, [previous_name, name], inplace=True)

        prev_previous_type = previous_type
        prev_previous_name = previous_name
        previous_type = type(module)
        previous_name = name