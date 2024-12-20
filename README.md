# MiDaS深度估计模型训练

本代码库包含了在DTU数据集上训练MiDaS（单目深度估计）模型的实现。该实现基于PyTorch框架，支持多种MiDaS模型架构。

## 环境准备

### 依赖包
```
torch
torchvision
opencv-python
numpy
pillow
matplotlib
```

### 数据集
- DTU数据集结构：
  ```
  DTU/
  ├── Rectified/         # 校正后的RGB图像
  │   └── scan*/
  │       └── *_r5000.png
  └── Depths/           # 深度图
      └── scan*_train/
          └── depth_map*.pfm
  ```
- 链接：https://aistudio.baidu.com/datasetdetail/207802/0
### 预训练模型
从MiDaS官方仓库下载预训练模型：
- DPT-SwinV2-T-256模型：[dpt_swin2_tiny_256.pt](https://github.com/isl-org/MiDaS)

## 项目结构
```
Train/
├── train.py           # 主训练脚本
├── MiDaS/            # MiDaS库文件
└── checkpoints/      # 模型检查点保存目录
```

## 训练配置

### 超参数设置
- 批次大小：4
- 学习率：1e-4
- 训练轮数：50
- 输入图像尺寸：256x256
- 损失函数：MSE损失
- 优化器：Adam

### 数据预处理
- 图像统一调整为256x256大小
- RGB图像标准化处理：
  - 均值：[0.485, 0.456, 0.406]
  - 标准差：[0.229, 0.224, 0.225]
- 深度图从PFM文件加载并调整尺寸

## 训练流程

1. **模型初始化**
   - 加载预训练的MiDaS模型
   - 支持从检查点恢复训练

2. **数据集加载**
   - 自动配对RGB图像和对应的深度图
   - 实时数据增强和转换
   - 支持多进程数据加载

3. **训练循环**
   - 每2个epoch保存检查点
   - 生成训练损失曲线图
   - 支持完整检查点和纯模型权重保存

## 模型检查点

训练过程会保存两种类型的检查点：
1. 完整检查点（`midas_checkpoint_epoch_*.pth`）：
   - 模型状态
   - 优化器状态
   - 当前轮数
   - 损失历史记录
2. 模型权重（`midas_model_epoch_*.pth`）：
   - 仅保存模型权重，用于推理

## 使用方法

1. 配置好目录结构并下载数据集
2. 将预训练模型放置在指定路径
3. 运行训练脚本：
   ```bash
   python train.py
   ```

## 训练监控

- 控制台实时显示训练进度
- 自动生成并保存损失曲线图
- 检查点保存在`checkpoints`目录

## 注意事项

- 代码自动选择CUDA或CPU设备
- 支持从上次检查点恢复训练
- 数据集加载器可以处理不匹配的图像/深度图对
- 每10个批次记录一次训练进度
- 确保数据集路径和预训练模型路径正确配置
