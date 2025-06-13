import torch 
import torch.nn as nn
from torch.nn import MSELoss
torch.random.manual_seed(1)
#这里直接创建数据来模拟了
yhat = torch.randn((50,),dtype=torch.float32)
y=torch.randn((50,),dtype=torch.float32)

#实例化损失函数
criterion=MSELoss( )
loss=criterion(yhat,y)

print(loss)