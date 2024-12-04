"""模型加载器模块
这个模块负责加载不同类型的MiDaS深度估计模型，包括：
1. DPT系列模型（基于不同的Transformer backbone）
2. 原始MiDaS模型
3. 轻量级MiDaS模型
4. OpenVINO优化的模型

每种模型都有其特定的预处理要求和网络配置。
"""

import cv2
import torch

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose

# 默认模型配置
# 键：模型类型标识符
# 值：对应的权重文件路径
default_models = {
    # DPT模型 - BEiT backbone
    "dpt_beit_large_512": "weights/dpt_beit_large_512.pt",     # BEiT-Large, 512x512
    "dpt_beit_large_384": "weights/dpt_beit_large_384.pt",     # BEiT-Large, 384x384
    "dpt_beit_base_384": "weights/dpt_beit_base_384.pt",       # BEiT-Base, 384x384
    
    # DPT模型 - Swin2 backbone
    "dpt_swin2_large_384": "weights/dpt_swin2_large_384.pt",   # Swin2-Large, 384x384
    "dpt_swin2_base_384": "weights/dpt_swin2_base_384.pt",     # Swin2-Base, 384x384
    "dpt_swin2_tiny_256": "weights/dpt_swin2_tiny_256.pt",     # Swin2-Tiny, 256x256
    
    # DPT模型 - 其他backbone
    "dpt_swin_large_384": "weights/dpt_swin_large_384.pt",     # Swin-Large, 384x384
    "dpt_next_vit_large_384": "weights/dpt_next_vit_large_384.pt",  # Next-ViT-Large, 384x384
    "dpt_levit_224": "weights/dpt_levit_224.pt",               # LeViT, 224x224
    "dpt_large_384": "weights/dpt_large_384.pt",               # ViT-Large, 384x384
    "dpt_hybrid_384": "weights/dpt_hybrid_384.pt",             # ViT-Hybrid, 384x384
    
    # 原始MiDaS模型
    "midas_v21_384": "weights/midas_v21_384.pt",               # MiDaS v2.1, 384x384
    "midas_v21_small_256": "weights/midas_v21_small_256.pt",   # MiDaS v2.1 Small, 256x256
    
    # OpenVINO优化模型
    "openvino_midas_v21_small_256": "weights/openvino_midas_v21_small_256.xml",  # OpenVINO优化版本
}


def load_model(device, model_path, model_type="dpt_large_384", optimize=True, height=None, square=False):
    """加载指定的深度估计网络
    
    Args:
        device (device): PyTorch设备对象（CPU或GPU）
        model_path (str): 模型权重文件的路径
        model_type (str): 模型类型，默认为"dpt_large_384"
        optimize (bool): 是否在CUDA上将模型优化为半精度，默认为True
        height (int): 推理时编码器的输入图像高度，默认为None
        square (bool): 是否将输入调整为正方形分辨率，默认为False
    
    Returns:
        tuple: (model, transform, net_w, net_h)
            - model: 加载的网络模型
            - transform: 用于预处理输入图像的转换函数
            - net_w: 网络输入宽度
            - net_h: 网络输入高度
            
    支持的模型类型：
    1. DPT系列（多种backbone）：
       - BEiT (Large/Base)
       - Swin2 (Large/Base/Tiny)
       - Swin
       - Next-ViT
       - LeViT
       - ViT (Large/Hybrid)
    2. MiDaS系列：
       - MiDaS v2.1
       - MiDaS v2.1 Small
    3. OpenVINO优化模型
    """
    if "openvino" in model_type:
        from openvino.runtime import Core

    keep_aspect_ratio = not square

    if model_type == "dpt_beit_large_512":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_512",
            non_negative=True,
        )
        net_w, net_h = 512, 512
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitb16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2l24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2b24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_tiny_256":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2t16_256",
            non_negative=True,
        )
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swinl12_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_next_vit_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="next_vit_large_6m",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_levit_224":
        model = DPTDepthModel(
            path=model_path,
            backbone="levit_384",
            non_negative=True,
            head_features_1=64,
            head_features_2=8,
        )
        net_w, net_h = 224, 224
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21_384":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small_256":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                               non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "openvino_midas_v21_small_256":
        ie = Core()
        uncompiled_model = ie.read_model(model=model_path)
        model = ie.compile_model(uncompiled_model, "CPU")
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    if not "openvino" in model_type:
        print("Model loaded, number of parameters = {:.0f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    else:
        print("Model loaded, optimized with OpenVINO")

    if "openvino" in model_type:
        keep_aspect_ratio = False

    if height is not None:
        net_w, net_h = height, height

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    if not "openvino" in model_type:
        model.eval()

    if optimize and (device == torch.device("cuda")):
        if not "openvino" in model_type:
            model = model.to(memory_format=torch.channels_last)
            model = model.half()
        else:
            print("Error: OpenVINO models are already optimized. No optimization to half-float possible.")
            exit()

    if not "openvino" in model_type:
        model.to(device)

    return model, transform, net_w, net_h
