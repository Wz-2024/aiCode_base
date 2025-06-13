import torch
import matplotlib.pyplot as plt
import numpy as np

def display_random_images(image_tensor, num_images=5):
    """
    从图像张量中随机抽取指定数量的图像并展示。
    
    参数:
        image_tensor (torch.Tensor): 形状为 [batch_size, channels, height, width] 的张量
        num_images (int): 要展示的图像数量，默认为5
    """
    # 检查输入张量的形状
    if len(image_tensor.shape) != 4:
        raise ValueError("输入张量必须是4维的 [batch_size, channels, height, width]")
    
    batch_size = image_tensor.shape[0]
    if num_images > batch_size:
        raise ValueError(f"要展示的图像数量 ({num_images}) 不能超过批次大小 ({batch_size})")
    
    # 随机选择5个图像的索引
    random_indices = torch.randperm(batch_size)[:num_images]
    selected_images = image_tensor[random_indices]  # 形状: [5, 3, 32, 32]

    # 将张量转换为适合 matplotlib 显示的格式
    # 1. 反归一化（假设输入是经过 Normalize([0.5], [0.5]) 处理的）
    selected_images = selected_images * 0.5 + 0.5  # 从 [-1, 1] 转换回 [0, 1]
    # 2. 转换为 numpy 并调整维度顺序: [5, 3, 32, 32] -> [5, 32, 32, 3]
    selected_images = selected_images.permute(0, 2, 3, 1).numpy()

    # 创建一个画布，展示5张图像
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(selected_images[i])
        ax.axis('off')  # 关闭坐标轴
        ax.set_title(f"Image {random_indices[i].item()}")

    plt.show()