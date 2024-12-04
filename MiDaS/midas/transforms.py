"""数据转换模块
这个模块包含了用于处理输入图像和深度图的各种转换操作，包括：
1. 图像大小调整（保持或不保持纵横比）
2. 图像标准化
3. 网络输入准备
"""

import numpy as np
import cv2
import math


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """调整样本大小，确保达到指定的最小尺寸，同时保持纵横比
    
    Args:
        sample (dict): 包含图像和深度图的样本字典
            - image: 输入图像
            - disparity: 视差/深度图
            - mask: 掩码
        size (tuple): 目标尺寸 (高度, 宽度)
        image_interpolation_method: 图像插值方法，默认使用INTER_AREA
    
    Returns:
        tuple: 调整后的尺寸 (高度, 宽度)
    """
    shape = list(sample["disparity"].shape)

    # 如果当前尺寸已经满足要求，直接返回
    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    # 计算缩放比例
    scale = [0, 0]
    scale[0] = size[0] / shape[0]  # 高度缩放比例
    scale[1] = size[1] / shape[1]  # 宽度缩放比例

    # 使用较大的缩放比例，确保两个维度都达到最小尺寸
    scale = max(scale)

    # 计算新的尺寸
    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # 调整图像大小
    sample["image"] = cv2.resize(
        sample["image"], 
        tuple(shape[::-1]),  # OpenCV需要(宽度, 高度)格式
        interpolation=image_interpolation_method
    )

    # 调整视差图大小（使用最近邻插值以保持深度值）
    sample["disparity"] = cv2.resize(
        sample["disparity"], 
        tuple(shape[::-1]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # 调整掩码大小
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """调整样本大小到指定的宽度和高度
    
    这个类提供了灵活的图像缩放选项：
    1. 可以选择是否保持纵横比
    2. 可以选择是否同时调整目标（深度图）
    3. 支持多种缩放策略
    4. 可以确保输出尺寸是指定数字的倍数
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """初始化Resize转换器
        
        Args:
            width (int): 目标宽度
            height (int): 目标高度
            resize_target (bool, optional): 是否调整目标大小
                True: 调整整个样本（图像、掩码、目标）
                False: 只调整图像
                默认为True
            keep_aspect_ratio (bool, optional): 是否保持纵横比
                True: 保持输入样本的纵横比
                输出可能不会完全符合指定的宽度和高度
                具体行为取决于resize_method参数
                默认为False
            ensure_multiple_of (int, optional): 确保输出维度是此数的倍数
                默认为1
            resize_method (str, optional): 缩放策略
                "lower_bound": 输出至少和指定尺寸一样大
                "upper_bound": 输出最大不超过指定尺寸
                "minimal": 尽可能少的缩放
                默认为"lower_bound"
            image_interpolation_method: 图像插值方法
                默认使用cv2.INTER_AREA
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        """确保数值是指定数的倍数
        
        Args:
            x: 输入数值
            min_val: 最小允许值
            max_val: 最大允许值
            
        Returns:
            int: 调整后的数值
        """
        # 四舍五入到最近的倍数
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        # 如果超过最大值，向下取整
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        # 如果小于最小值，向上取整
        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    """图像标准化处理
    
    将图像按照给定的均值和标准差进行标准化：
    normalized = (image - mean) / std
    """

    def __init__(self, mean, std):
        """初始化图像标准化器
        
        Args:
            mean: 各通道的均值
            std: 各通道的标准差
        """
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        """执行标准化操作
        
        Args:
            sample (dict): 包含图像的样本字典
            
        Returns:
            dict: 标准化后的样本
        """
        sample["image"] = (sample["image"] - self.__mean) / self.__std
        return sample


class PrepareForNet(object):
    """准备网络输入数据
    
    执行以下操作：
    1. 将图像转换为torch张量格式
    2. 调整维度顺序为网络所需的格式
    3. 将图像值范围调整到[0,1]
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        """准备网络输入
        
        Args:
            sample (dict): 包含图像的样本字典
            
        Returns:
            dict: 处理后的样本，可直接输入网络
        """
        image = np.ascontiguousarray(sample["image"]).astype(np.float32)
        
        # 确保图像值在[0,1]范围内
        if (image.max() - image.min()) > 1e-6:
            image = (image - image.min()) / (image.max() - image.min())
        
        # 调整维度顺序：HWC -> CHW
        image = image.transpose(2, 0, 1)
        
        sample["image"] = image
        return sample
