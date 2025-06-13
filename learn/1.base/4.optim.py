import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


torch.manual_seed(1)
X=torch.rand((500,20),dtype=torch.float32)
y=torch.randint(0,3,(500,),dtype=torch.float32)

input_=X.shape[1]
output_=len(y.unique())

#定义架构
class Model(nn.Module):
    def __init__(self,in_features=10,out_features=3):
        super().__init__()
        self.linear1=nn.Linear(in_features,13)
        self.linear2=nn.Linear(13,8)
        self.output=nn.Linear(8,out_features)
    
    def forward(self,x):
        z1=self.linear1(x)
        sigma=F.relu(z1)
        z2=self.linear2(sigma)
        sigma=F.sigmoid(z2)
        z3=self.output(sigma)#这一步是linear,没有激活
        return z3
    
#实例化神经网络
torch.manual_seed(112)
model=Model(in_features=input,out_features=output_)

#定义损失函数
criterion=nn.CrossEntropyLoss()



#定义优化算法
#确定一些超参数
lr=0.1
gamma=0.9
#表示学习率和动量
optimizer=optim.SGD(model.parameters(),lr=lr,momentum=gamma)

#接下来开始一轮梯度下降
zhat=model.forward(X)
loss=criterion(zhat,y)
loss.backward()
optimizer.step()#更新权重w,此时坐标就进行了一次迭代,所有的梯度开始重新计算
optimizer.zero_grad()#清除之前的缓存,基于上一次迭代开始为下一次计算做准备  不写这句话很可能导致错误的参数迭代

print(loss)
