"""
MiDaS基础模型类
这个文件定义了所有MiDaS模型的基类，提供基本的模型加载功能
"""

import torch


class BaseModel(torch.nn.Module):
    """
    MiDaS基础模型类，继承自PyTorch的nn.Module
    作为其他所有MiDaS模型的父类，提供基础功能
    """
    
    def load(self, path):
        """从文件加载模型参数

        Args:
            path (str): 模型权重文件的路径
        
        说明:
            1. 首先将模型加载到CPU设备上
            2. 如果加载的是检查点文件（包含优化器状态），则只提取模型参数
            3. 使用load_state_dict将参数加载到当前模型中
        """
        # 加载模型参数到CPU设备
        parameters = torch.load(path, map_location=torch.device('cpu'))

        # 如果是检查点文件（包含优化器状态），只提取模型参数
        if "optimizer" in parameters:
            parameters = parameters["model"]

        # 将参数加载到当前模型
        self.load_state_dict(parameters)
