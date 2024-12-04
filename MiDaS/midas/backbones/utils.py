"""Backbone工具模块

这个模块提供了一系列用于处理Vision Transformer (ViT)类backbone的工具类和函数：
1. 特征切片和处理类（Slice, AddReadout, ProjectReadout）
2. 维度转换类（Transpose）
3. 特征提取和处理函数
4. Backbone构建函数

主要用于：
1. 处理ViT的token特征
2. 构建和配置backbone网络
3. 提取和后处理多尺度特征
"""

import torch
import torch.nn as nn


class Slice(nn.Module):
    """特征切片类
    
    从输入张量中提取指定索引之后的所有特征。
    主要用于去除ViT中的[CLS]等特殊token。
    """
    def __init__(self, start_index=1):
        """
        Args:
            start_index (int): 开始切片的索引位置，默认为1（跳过[CLS]token）
        """
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):
    """Readout添加类
    
    将readout token（通常是[CLS]token）添加到其他token特征上。
    可以选择使用单个readout或两个readout的平均值。
    """
    def __init__(self, start_index=1):
        """
        Args:
            start_index (int): readout token之后的起始索引，默认为1
        """
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            # 如果有两个readout token，取平均值
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            # 否则使用第一个token
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    """Readout投影类
    
    将readout token投影到特征空间，并与其他token特征融合。
    使用线性投影和GELU激活函数。
    """
    def __init__(self, in_features, start_index=1):
        """
        Args:
            in_features (int): 输入特征维度
            start_index (int): readout token之后的起始索引，默认为1
        """
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(
            nn.Linear(2 * in_features, in_features),
            nn.GELU()
        )

    def forward(self, x):
        # 扩展readout token并与特征拼接
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    """维度转置类
    
    转置张量的指定维度。用于调整特征图的形状。
    """
    def __init__(self, dim0, dim1):
        """
        Args:
            dim0 (int): 第一个要转置的维度
            dim1 (int): 第二个要转置的维度
        """
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


# 用于存储中间激活值的字典
activations = {}


def get_activation(name):
    """获取激活值的钩子函数
    
    Args:
        name (str): 激活值的标识名
        
    Returns:
        function: 钩子函数，用于获取并存储指定层的输出
    """
    def hook(model, input, output):
        activations[name] = output

    return hook


def forward_default(pretrained, x, function_name="forward_features"):
    """默认的前向传播函数
    
    执行预训练模型的特征提取，并获取多个层的输出。
    
    Args:
        pretrained: 预训练模型
        x: 输入数据
        function_name (str): 要执行的函数名，默认为"forward_features"
        
    Returns:
        tuple: (layer_1, layer_2, layer_3, layer_4) 四个层的输出特征
    """
    # 执行特征提取
    exec(f"pretrained.model.{function_name}(x)")

    # 获取各层激活值
    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    # 后处理（如果有）
    if hasattr(pretrained, "act_postprocess1"):
        layer_1 = pretrained.act_postprocess1(layer_1)
    if hasattr(pretrained, "act_postprocess2"):
        layer_2 = pretrained.act_postprocess2(layer_2)
    if hasattr(pretrained, "act_postprocess3"):
        layer_3 = pretrained.act_postprocess3(layer_3)
    if hasattr(pretrained, "act_postprocess4"):
        layer_4 = pretrained.act_postprocess4(layer_4)

    return layer_1, layer_2, layer_3, layer_4


def forward_adapted_unflatten(pretrained, x, function_name="forward_features"):
    """适配的前向传播函数，带展开操作
    
    执行特征提取并将扁平化的特征重新展开为2D特征图。
    
    Args:
        pretrained: 预训练模型
        x: 输入数据
        function_name (str): 要执行的函数名，默认为"forward_features"
        
    Returns:
        tuple: (layer_1, layer_2, layer_3, layer_4) 四个层的输出特征
    """
    # 记录输入形状
    b, c, h, w = x.shape

    # 执行特征提取
    exec(f"glob = pretrained.model.{function_name}(x)")

    # 获取各层激活值
    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    # 部分后处理
    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    # 展开操作
    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3: len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3: len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3: len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3: len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    """获取Readout操作
    
    根据use_readout参数选择不同的Readout操作。
    
    Args:
        vit_features (int): ViT特征维度
        features (list): 输出特征维度列表
        use_readout (str): Readout操作类型，可以是"ignore", "add", "project"
        start_index (int): Readout token之后的起始索引，默认为1
        
    Returns:
        list: Readout操作列表
    """
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


def make_backbone_default(
        model,
        features=[96, 192, 384, 768],
        size=[384, 384],
        hooks=[2, 5, 8, 11],
        vit_features=768,
        use_readout="ignore",
        start_index=1,
        start_index_readout=1,
):
    """构建默认的Backbone
    
    构建一个预训练模型的Backbone，并配置Readout操作和后处理函数。
    
    Args:
        model: 预训练模型
        features (list): 输出特征维度列表，默认为[96, 192, 384, 768]
        size (list): 输入图像大小，默认为[384, 384]
        hooks (list): Hook点索引列表，默认为[2, 5, 8, 11]
        vit_features (int): ViT特征维度，默认为768
        use_readout (str): Readout操作类型，默认为"ignore"
        start_index (int): Readout token之后的起始索引，默认为1
        start_index_readout (int): Readout token之后的起始索引，默认为1
        
    Returns:
        nn.Module: 构建好的Backbone模型
    """
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index_readout)

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    return pretrained
