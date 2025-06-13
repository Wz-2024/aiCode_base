import torch
import torch.nn as nn

# 创建一个线性层，输入维度为2，输出维度为1
linear_layer = nn.Linear(2, 1)

# 定义输入数据张量 X
X = torch.tensor([[0,0], [1,0], [0,1], [1,1]], dtype=torch.float32)

# 将输入数据传递给线性层，得到输出
output = linear_layer(X)

print(output)