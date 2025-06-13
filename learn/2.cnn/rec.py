import torch
from torch import nn

# 假设一组数据
# 图像数据的格式 samples,channels,height,width
data = torch.ones(size=(10, 3, 28, 28))

"""
    构建卷积层,当然这种写法有点啰嗦,一般直接写成
    conv=nn.Conv2d(3,6,4) 表示in_channels=3,out_channels=6,kernel_size=4
"""
conv1 = nn.Conv2d(
    in_channels=3,  # 表示输入的通道数
    out_channels=6,  # 表示输出的通道数,,这一位者图像中的每个通道都会被扫描6次
    # 所有的RGB通道都会被扫描六次
    kernel_size=3,  # 表示卷积核的尺寸，3*3
    stride=1,  # 表示步长
    padding=1,  # 表示填充
)
conv2 = nn.Conv2d(
    in_channels=6,  # 表示输入的通道数
    out_channels=4,
    kernel_size=3,  # 表示卷积核的尺寸，3*3
    stride=1,  # 表示步长
    padding=1,  # 表示填充
)

"如果当前没有填充,那么输出的维度为(10,6,26,26),"
data = conv1(data)
print(data.shape)
data = conv2(data)
print(data.shape)
